import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
import llava.model.blip_llama_infer as modeling_llama
from PIL import Image
import json
import os
import requests
from PIL import Image
from io import BytesIO

from peft import PeftModel
from tqdm.auto import tqdm
from llava.model.blip2 import Blip2Base 

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
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
    
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    lora_weight = os.path.expanduser(args.lora_weight)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
    model = modeling_llama.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(args.device)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    print("mm_use_im", mm_use_im_start_end)
    # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    image_token_len = 32  
    
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = model.model.visual_encoder.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    model.model.visual_encoder.config = vision_config
    
    model.to('cpu')  # lora weight is on cpu
    model = PeftModel.from_pretrained(
            model,
            lora_weight,
            torch_dtype=torch.bfloat16,
        )
    model.to(args.device)
    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)
        
    model.to(args.device)

    with open('/home/shawn/nvme/vl_research/peft_llama/data/OwlEval/questions.jsonl', 'r') as json_file:
        questions = list(json_file)
        
    #questions = json.load(open("/home/shawn/nvme/vl_research/peft_llama/data/OwlEval/questions.jsonl", "r"))
    #questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    image_folder = "/home/shawn/nvme/vl_research/peft_llama/data/OwlEval/cases/"
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    
    for file in tqdm(questions):
        
        line = json.loads(file)
        
        idx = line["question_id"]
        image_name = line["image"]
        ques = line["question"]
        text_input = line["question"]
            
        if mm_use_im_start_end:
            qs = ques + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = ques + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        
        image = load_image(os.path.join(image_folder, image_name))
        from transformers import CLIPImageProcessor, BlipImageProcessor
        image_processor =  BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        #image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
        image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=args.device)
        
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
        
        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                text_input=text_input,
                images=image_tensor,
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
        
        ans_file.write(json.dumps({"image": image_name,
                                   "question_id": idx,
                                   "question": ques,
                                   "answer": outputs,
                                   "model_id": model_name}) + "\n")
        ans_file.flush()
    ans_file.close() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Path to the model")
    parser.add_argument("--lora-weight", type=str, required=True, help="Path to the lora weight")
    # parser.add_argument("--mm-projector", type=str, default= "data/LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin")     #"data/mm_projector/llava-7b-pretrain.bin") # this is stage 2 
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--conv-mode", type=str, default="multimodal")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--device", type=str,default="cuda:0")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
