import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
import llava.model.blip_llama_infer as blip_llama
from llava.model.blip2 import Blip2Base, disabled_train
from llava.model.dist_utils import download_cached_file

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    model = blip_llama.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(args.device)
    #print(model.model.Qformer.dtype)
    #model.model.Qformer = model.model.Qformer.to(dtype=torch.float32)
    #-------------------------------------------------------
    # qformer_tokenizer = Blip2Base.init_tokenizer(truncation_side="left")
    # #model.model.load_state_dict(torch.load(lora_weight, map_location='cpu'))
    # q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
    # cached_file = download_cached_file(
    #         q_former_model, check_hash=False, progress=True
    #     )
    # q_former_checkpoint = torch.load(cached_file, map_location="cpu")
    # q_former_state_dict = q_former_checkpoint["model"]

    # vision_tower, ln_vision =  Blip2Base.init_vision_encoder(
    #         model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision="fp16" ## hack changed it to fp16
    #     )
    # freeze_vit = True # freeze vision encoder

    # for name, param in vision_tower.named_parameters():
    #     param.requires_grad = False
    # vision_tower.eval()
    # vision_tower.train = disabled_train

    # ln_vision_weight = q_former_state_dict['ln_vision.weight']
    # ln_vision_bias = q_former_state_dict['ln_vision.bias']
    # ln_vision.weight = torch.nn.Parameter(ln_vision_weight).to(torch.float32)
    # ln_vision.bias = torch.nn.Parameter(ln_vision_bias).to(torch.float32)
    # for name, param in ln_vision.named_parameters():
    #     param.requires_grad = False
    # ln_vision.eval()
    # ln_vision.train = disabled_train

    # Qformer, query_tokens = Blip2Base.init_Qformer(
    #     num_query_token=32, vision_width=model.model.visual_encoder.num_features )
    
    # Qformer.load_state_dict(q_former_state_dict, strict=False)
    # Qformer.resize_token_embeddings(len(qformer_tokenizer))
    
    # Qformer.cls = None
    # q = q_former_state_dict['query_tokens'].clone().detach().to(dtype=torch.float16) #.requires_grad_(True)
    # #q = torch.tensor(q_former_state_dict['query_tokens'], dtype=torch.float16)
    # model.model.query_tokens = torch.nn.Parameter(q)     
    # model.model.Qformer = Qformer.half()   
    # model.model.visual_encoder = vision_tower
    # model.model.ln_vision = ln_vision   #.half().cuda()
    # model.model.tokenizer = qformer_tokenizer
    # -------------------------------------------------------
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    print("mm_use_im", mm_use_im_start_end)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    image_token_len = 32  
    #todo add pad token if for batched inference 
    
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = model.model.visual_encoder.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    model.model.visual_encoder.config = vision_config
    # # mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
    # # mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
    # # mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

    # # model.model.mm_projector = mm_projector.cuda().half()
    # # model.model.vision_tower = [vision_tower]
    # Qformer, query_tokens = Blip2Base.init_Qformer(
    #         num_query_token=32, vision_width=model.model.visual_encoder.num_features )
    
    # q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
    # cached_file = download_cached_file(
    #         q_former_model, check_hash=False, progress=True
    #     )
    # q_former_checkpoint = torch.load(cached_file, map_location="cpu")
    # q_former_state_dict = q_former_checkpoint["model"]
    # Qformer.load_state_dict(q_former_state_dict, strict=False)
    # print("\nLoaded trainable pretrained Qformer weights")

    # Qformer.cls = None
    # Qformer.bert.embeddings.word_embeddings = None
    # Qformer.bert.embeddings.position_embeddings = None
    # for layer in Qformer.bert.encoder.layer:
    #     layer.output = None
    #     layer.intermediate = None
    # model.model.query_tokens = torch.nn.Parameter(torch.FloatTensor(query_tokens).half().to(device='cuda'))

    # model.model.Qformer = Qformer.half().cuda()
    # mm_projector =  torch.nn.Linear(
    #         model.model.Qformer.config.hidden_size, model.model.config.hidden_size
    #     )   #model.model.llama_proj
    # mm_projector_weights = torch.load("data/minigpt_proj_7b.pth", map_location='cpu')['model']
    # mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if 'llama_proj' == k.split('.')[0]})

    # model.model.llama_proj = mm_projector.half().cuda()
    # print("\nLoaded pretrained mm_projector")
    
    model.to(args.device)
    
    qs = args.query
    text_input = args.query
    if mm_use_im_start_end:
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    input_ids = torch.as_tensor(inputs.input_ids).to(args.device) # , dtype=torch.float16).cuda()  ## added dtype
    # --------- -----
    # image2 = load_image("https://llava-vl.github.io/static/images/view.jpg")
    # qs2 = "What are the things I should be cautious about when I visit here?"
    # image_tensor_2 = image_processor.preprocess(image2, return_tensors='pt')['pixel_values'][0]
    # image_input = torch.stack([image_tensor, image_tensor_2]).to(args.device)
    # text_input = [text_input, qs2]
    
    # ------ --- -- - - -- - -
    
    keywords = ['###']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    #model = model.eval()
    print("input_ids", input_ids)
    #print("image_tensor", image_tensor.unsqueeze(0).half().cuda())
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            text_input=text_input, 
            images= image_tensor.unsqueeze(0).to(args.device),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    print("outputs", outputs)

    while True:
        cur_len = len(outputs)
        outputs = outputs.strip()
        for pattern in ['###', 'Assistant:', 'Response:']:
            if outputs.startswith(pattern):
                outputs = outputs[len(pattern):].strip()
        if len(outputs) == cur_len:
            break

    try:
        index = outputs.index(conv.sep)
    except ValueError:
        outputs += conv.sep
        index = outputs.index(conv.sep)

    outputs = outputs[:index].strip()
    print("\n", outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--conv-mode", type=str, default="multimodal")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()

    eval_model(args)
