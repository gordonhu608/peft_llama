# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch

import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from llava import conversation as conversation_lib

from PIL import Image
import torch.nn as nn

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from llava.model.blip2 import Blip2Base, disabled_train
import llava.model.blip_llama as modeling_llama
from llava.model.dist_utils import download_cached_file

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_MASK_PATCH_TOKEN = "<mask_patch>"
DEFAULT_MASK_START_TOKEN = "<mask_start>"
DEFAULT_MASK_END_TOKEN = "<mask_end>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    maskmodel: bool = field(default=False)
    qformer_text_input: bool = field(default=True) 

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    
    #hack q_former text input 
    
    def strip_word_from_string(sentence, word):
        if sentence.startswith(word):
            sentence = sentence[len(word):].lstrip()
        if sentence.endswith(word):
            sentence = sentence[:-len(word)].rstrip()
        return sentence

    qformer_text_input = []
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        temp = "".join([strip_word_from_string(s['value'], '<image>') for s in source if s["from"] == "human"])
        qformer_text_input.append(temp)
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
    #print("check qformer text input length = 1", len(qformer_text_input))
    return dict(input_ids=input_ids, labels=targets, qformer_text_input=qformer_text_input)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.multimodal_cfg['image_folder']
            processor = self.multimodal_cfg['image_processor']
            image = Image.open(os.path.join(image_folder, image_file))
            if self.multimodal_cfg['image_aspect_ratio'] == 'keep':
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            cur_token_len = 32
            cur_mask_len = 256 
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.multimodal_cfg, cur_token_len, cur_mask_len)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             qformer_text_input=data_dict["qformer_text_input"][0])
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.multimodal_cfg['is_multimodal']:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        text_input = [instance['qformer_text_input'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            text_input=text_input,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                multimodal_cfg=dict(
                                    is_multimodal=data_args.is_multimodal,
                                    image_token_len=data_args.image_token_len,
                                    image_folder=data_args.image_folder,
                                    image_aspect_ratio=data_args.image_aspect_ratio,
                                    use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                                    image_processor=getattr(data_args, 'image_processor', None)))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    #todo may need to consider not loading to cpu first if using multiple gpus 
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        #gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    #device = torch.device("cuda") #("cpu") #("cuda:0")
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        #device_map=device_map,
    )

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    assert tokenizer.pad_token == DEFAULT_PAD_TOKEN
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    
    assert tokenizer.eos_token == DEFAULT_EOS_TOKEN
    assert tokenizer.bos_token == DEFAULT_BOS_TOKEN
    assert tokenizer.unk_token == DEFAULT_UNK_TOKEN
    # if "llama" in model_args.model_name_or_path:
    #     tokenizer.add_special_tokens({
    #         "eos_token": DEFAULT_EOS_TOKEN,
    #         "bos_token": DEFAULT_BOS_TOKEN,
    #         "unk_token": DEFAULT_UNK_TOKEN,
    #     })
    if model_args.vision_tower is not None:
        #model.config.mm_vision_tower = model_args.vision_tower

        from transformers import CLIPVisionModel, CLIPImageProcessor
        dtype = torch.float32
        if training_args.fp16:
            dtype = torch.float16
        if training_args.bf16:
            dtype = torch.bfloat16

        # if not hasattr(model.model, 'vision_tower'):
        #     vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower)
        # else:
        #     vision_tower = model.model.vision_tower[0]
        vision_tower, _ =   Blip2Base.init_vision_encoder(
            model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision="fp16"
        )
        # for name, param in vision_tower.named_parameters():
        #     param.requires_grad = False
        # vision_tower.eval()
        # vision_tower.train = disabled_train
        # for name, param in model.model.ln_vision.named_parameters():
        #     param.requires_grad = False
        # model.model.ln_vision.eval()
        # model.model.ln_vision.train = disabled_train
        # print("freeze vision encoder") # logging 
        # vision_tower.requires_grad_(False)
        
        # vision_tower.to(dtype=dtype) #, device=model.model.visual_encoder.device)
        # model.model.visual_encoder = vision_tower    
        # print("\nLoading the pretrained vision encoder")
            
        image_processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)

        vision_config = vision_tower.config
        #todo start here flexible to both image and mask also edit in the dataset and preprocess token len
        #todo 
        #todo 
        #todo   
        num_patches = (32, 256)
        data_args.image_token_len = 32
        data_args.mask_token_len = 256
        data_args.image_processor = image_processor
        data_args.is_multimodal = True
        #--------------------------------------
        # q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
        # cached_file = download_cached_file(
        #         q_former_model, check_hash=False, progress=True
        #     )
        # q_former_checkpoint = torch.load(cached_file, map_location="cpu")
        # q_former_state_dict = q_former_checkpoint["model"]
        # msg = model.model.Qformer.load_state_dict(q_former_state_dict, strict=False)
        # #logger.info("Missing keys {}".format(msg.missing_keys))
        # logging.info("load checkpoint Qformer from %s" % q_former_model)
        # print("\nLoaded trainable pretrained Qformer weights")
        #--------------------------------------
        # model.config.use_mm_proj = True
        # model.config.mm_hidden_size = vision_config.hidden_size
        # model.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        # if not hasattr(model.model, 'mm_projector'):
        #     mm_projector = nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        # else:
        #--------------------------------------
        # mm_projector = model.model.llama_proj

        # if model_args.pretrain_mm_mlp_adapter is not None:
        #     mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')['model']
        #     mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if 'llama_proj' == k.split('.')[0]})

        # model.model.llama_proj = mm_projector
        # print("\nLoaded pretrained mm_projector")
        #--------------------------------------
        # model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        # if model_args.tune_mm_mlp_adapter:
        #     model.requires_grad_(False)
        #     for p in mm_projector.parameters():
        #         p.requires_grad = True

        print("model_args.mm_use_im_start_end: ", model_args.mm_use_im_start_end)
        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        #tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        print("tokenizer vocab size: ", len(tokenizer))
        vision_config.use_im_start_end = model_args.mm_use_im_start_end
        if model_args.mm_use_im_start_end:
            #num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            #model.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
            num_new_tokens = 0 
            # if num_new_tokens > 0:
            #     input_embeddings = model.get_input_embeddings().weight.data
            #     output_embeddings = model.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # if model_args.tune_mm_mlp_adapter:
            #     model.model.orig_embeds_params = [model.get_input_embeddings().weight.data.clone().to(device=training_args.device)] # training_args.device
            #     for p in model.get_input_embeddings().parameters():
            #         p.requires_grad = True
            #     for p in model.get_output_embeddings().parameters():
            #         p.requires_grad = False

            # if model_args.pretrain_mm_mlp_adapter:
            #     mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            #     embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            #     assert num_new_tokens == 2
            #     if input_embeddings.shape == embed_tokens_weight.shape:
            #         input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            #     elif embed_tokens_weight.shape[0] == num_new_tokens:
            #         input_embeddings[-num_new_tokens:] = embed_tokens_weight
            #     else:
            #         raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

        model.model.visual_encoder.config = vision_config

    #hack added peft 
    #model = prepare_model_for_int8_training(model)

    # if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    # else:
    #     def make_inputs_require_grad(module, input, output):
    #         output.requires_grad_(True)

    #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    for param in model.parameters():
        # freeze base model's layers
        param.requires_grad = False
    
    model.enable_input_require_grads()
    
    config = LoraConfig(
        r= 8, #lora_r, #16
        lora_alpha=32, #lora_alpha,
        target_modules=  ["q_proj", "v_proj"], #['q_proj','k_proj','v_proj','o_proj'], # , # lora_target_modules,
        lora_dropout=0.05,  
        bias="none",
        task_type="CAUSAL_LM",
        #modules_to_save=['Qformer', 'llama_proj', 'query_tokens'] #hack 
    )
    
    model = get_peft_model(model, config)

    model.print_trainable_parameters() 
    
    if training_args.bf16:
        print("convert model to bf16")
        model = model.bfloat16()
    else:
        model = model.float()
    
    #model.train()

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" :
    #     model = torch.compile(model)


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)

    model.save_pretrained('save_pretrained/stage2_only_llm_blip_instruct')

if __name__ == "__main__":
    print("This is from peft_llama script")
    train()