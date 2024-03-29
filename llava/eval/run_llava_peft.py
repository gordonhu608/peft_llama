import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
import llava.model.blip_llama_infer as blip_llama
from llava.model.blip2 import Blip2Base
from llava.model.dist_utils import download_cached_file
from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from peft import PeftModel

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
    lora_weight = os.path.expanduser(args.lora_weight)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # if args.mm_projector is None:
    #     model = modeling_llama.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    #     image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    #mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #     if mm_use_im_start_end:
    #         tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    #     vision_tower = model.model.vision_tower[0]
    #     vision_tower.to(device='cuda', dtype=torch.float16)
    #     vision_config = vision_tower.config
    #     vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    #     vision_config.use_im_start_end = mm_use_im_start_end
    #     if mm_use_im_start_end:
    #         vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    #     image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    # else:
        # in case of using a pretrained model with only a MLP projector weights
    model = blip_llama.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

    #model.model.load_state_dict(torch.load(lora_weight, map_location='cpu'))

    vision_tower, _ =  Blip2Base.init_vision_encoder(
            model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision="fp16"
        )
    
        #  self.visual_encoder, self.ln_vision = Blip2Base.init_vision_encoder(
        #     model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision="fp16"
        # )
    #CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.bfloat16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert len(tokenizer) == 32004
    #tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #todo add pad token if for batched inference 
    
    #model.model.visual_encoder = vision_tower
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        print("using im start end")
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    image_token_len = 32  #(vision_config.image_size // vision_config.patch_size) ** 2
    
    model.model.visual_encoder.config = vision_config
    # mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
    # mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
    # mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

    # model.model.mm_projector = mm_projector.cuda().half()
    # model.model.vision_tower = [vision_tower]
    
    # q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
    # cached_file = download_cached_file(
    #         q_former_model, check_hash=False, progress=True
    #     )
    # q_former_checkpoint = torch.load(cached_file, map_location="cpu")
    # q_former_state_dict = q_former_checkpoint["model"]
    # msg = model.model.Qformer.load_state_dict(q_former_state_dict, strict=False)
    # print("\nLoaded trainable pretrained Qformer weights")
    
    # mm_projector = model.model.llama_proj
    # mm_projector_weights = torch.load("data/minigpt_proj_7b.pth", map_location='cpu')['model']
    # mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items() if 'llama_proj' == k.split('.')[0]})

    # model.model.llama_proj = mm_projector
    # print("\nLoaded pretrained mm_projector")
    
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
    input_ids = torch.as_tensor(inputs.input_ids).cuda() # , dtype=torch.float16).cuda()  ## added dtype

    keywords = ['###']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    model.to('cpu') # lora weight is on cpu
    model = PeftModel.from_pretrained(
            model,
            lora_weight,
            torch_dtype=torch.bfloat16,
        )
    model.to('cuda')
    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("input_ids", input_ids)
    print("max`", input_ids.max())
    print("min", input_ids.min())
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            text_input=text_input,
            images=image_tensor.unsqueeze(0).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

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
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument("--conv-mode", type=str, default="multimodal")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--lora-weight", type=str)
    args = parser.parse_args()

    eval_model(args)
