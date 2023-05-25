import argparse
#from models.BLIP2.BLIP2 import BLIP2
import more_itertools
from tqdm.auto import tqdm
import datetime
import os
import json
import re
from datasets.vqa_dataset import textVQADataset, docVQADataset, ocrVQADataset, STVQADataset
from datasets.ocr_dataset import ocrDataset
#from models.lavis.lavis import lavis
import torch
import numpy as np

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
from llava.utils import KeywordsStoppingCriteria, load_image

from peft import PeftModel

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def get_model(args):
    disable_torch_init()
    dtype = torch.bfloat16
    model_name = os.path.expanduser(args.model_name)
    lora_weight = os.path.expanduser(args.lora_weight)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    model = blip_llama.LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda() 

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    print("mm_use_im", mm_use_im_start_end)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    image_token_len = 32  
    
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = model.model.visual_encoder.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    model.model.visual_encoder.config = vision_config
    #hack add instruct blip 

    instruct_qformer = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"
    cached_file = download_cached_file(
                instruct_qformer, check_hash=False, progress=True
            )
    q_former_checkpoint = torch.load(cached_file, map_location="cpu")
    q_former_state_dict = q_former_checkpoint["model"]
    
    # for name, param in q_former_state_dict.items():
    #     print(name, param.size())
    # import sys 
    # sys.exit(1)
    
    vision_tower, ln_vision =   Blip2Base.init_vision_encoder(
        model_name="eva_clip_g", img_size=224, drop_path_rate=0, use_grad_checkpoint=False, precision="fp16"
    )

    ln_vision_weight = q_former_state_dict['ln_vision.weight']
    ln_vision_bias = q_former_state_dict['ln_vision.bias']
    ln_vision.weight = torch.nn.Parameter(ln_vision_weight)
    ln_vision.bias = torch.nn.Parameter(ln_vision_bias)
    for name, param in ln_vision.named_parameters():
        param.requires_grad = False
    ln_vision.eval()
    ln_vision.train = disabled_train

    ln_vision.to(dtype=dtype, device=args.device)   
    model.model.ln_vision = ln_vision
    print("\nLoading the pretrained vision encoder")
    
    Qformer, query_tokens = Blip2Base.init_Qformer(
            num_query_token=32, vision_width=model.model.visual_encoder.num_features )
    Qformer.load_state_dict(q_former_state_dict, strict=False)
    
    qformer_tokenizer = Blip2Base.init_tokenizer(truncation_side="left")  

    Qformer.resize_token_embeddings(len(qformer_tokenizer))
    
    Qformer.cls = None
    
    llm_proj_weight = q_former_state_dict['llm_proj.weight']
    llm_proj_bias = q_former_state_dict['llm_proj.bias']
    llama_proj = model.model.llama_proj
    llama_proj.weight = torch.nn.Parameter(llm_proj_weight)
    llama_proj.bias = torch.nn.Parameter(llm_proj_bias)
    for name, param in llama_proj.named_parameters():
        param.requires_grad = False
    llama_proj.eval()
    llama_proj.train = disabled_train
    llama_proj.to(dtype=dtype, device=args.device)
    model.model.llama_proj = llama_proj
    
    model.model.query_tokens = torch.nn.Parameter(q_former_state_dict['query_tokens'].to(dtype=dtype))
    model.model.Qformer = Qformer.to(dtype=dtype)
    model.model.tokenizer = qformer_tokenizer
    # model.to('cpu')  # lora weight is on cpu
    # model = PeftModel.from_pretrained(
    #         model,
    #         lora_weight,
    #         torch_dtype=torch.bfloat16,
    #     )
    # model.to(args.device)
    # model.eval()
    # if torch.__version__ >= "2":
    #     model = torch.compile(model)
    model.to(args.device)
    return model, tokenizer

def has_word(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence)
    if match:
        return True
    else:
        return False
def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s
class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, answer, gt_answers):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if type(gt_answers)==list:
            for i in range(len(gt_answers)):
                gt_answers[i] = gt_answers[i].replace("\n", " ")
                gt_answers[i] = gt_answers[i].replace("\t", " ")
                gt_answers[i] = gt_answers[i].strip()
                gt_answers[i] = self.processPunctuation(gt_answers[i])
                gt_answers[i] = self.processDigitArticle(gt_answers[i])
                if has_word(answer, gt_answers[i]):
                    return 1
            return 0
        else:
            gt_answers = gt_answers.replace("\n", " ")
            gt_answers= gt_answers.replace("\t", " ")
            gt_answers = gt_answers.strip()
            gt_answers = self.processPunctuation(gt_answers)
            gt_answers = self.processDigitArticle(gt_answers)
            if has_word(answer, gt_answers[i]):
                return 1
            else:
                return 0

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText
def evaluate_VQA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    tokenizer,
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    #hack 
    #img_list = []
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        #hack change peft vl model generate
       
        qs = batch['question']
        text_input = batch['question']
        mm_use_im_start_end = True 
        image_token_len = 32  
        if mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        image = load_image(batch['image_path'])
        # img_list.append(batch['image_path'])
        # print("\nquestion: " ,batch['question'])
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_ids = torch.as_tensor(inputs.input_ids).to(args.device) # , dtype=torch.float16).cuda()  ## added dtype

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                text_input=text_input, 
                images=image_tensor.unsqueeze(0).to(args.device),
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
        print("\n outputs:", outputs, "\n answer", batch['gt_answers'])
        #output = #model.generate(image=batch['image_path'], question=batch['question'])
        answer_dict={'question':batch['question'], 'answer':outputs, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    eval = VQAEval()
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer,gt_answers)==1:
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    
    #print(img_list)
    
    return float(correct)/num
def evaluate_OCR(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    question='what is written in the image?',
    batch_size=1,
    answer_path='./answers'
):
    predictions=[]
    for batch in more_itertools.chunked(
        tqdm(dataset, desc="Running inference"), batch_size
    ):
        batch = batch[0]
        output = model.generate(image=batch['image_path'], question=question)
        answer_dict={'question':question, 'answer':output, 
        'gt_answers':batch['gt_answers'], 'image_path':batch['image_path'],
        'model_name':model_name}
        predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            gt_answers = remove_special_chars(gt_answers).lower()
            answer = remove_special_chars(answer).lower()
            if has_word(answer, gt_answers):
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num
            

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    #OCR datasets
    parser.add_argument("--ocr_dir_path", type=str, default="./data")
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K svt IC13_857 IC15_1811 svtp ct80 cocotext ctw totaltext HOST WOST WordArt")
    #textVQA
    parser.add_argument("--textVQA_image_dir_path", type=str, default="../data/textVQA/train_images")
    parser.add_argument("--textVQA_ann_path", type=str, default="../data/textVQA/TextVQA_0.5.1_val.json")

    #docVQA
    parser.add_argument("--docVQA_image_dir_path", type=str, default="./data/docVQA/val")
    parser.add_argument("--docVQA_ann_path", type=str, default="./data/docVQA/val/val_v1.0.json")

    #ocrVQA
    parser.add_argument("--ocrVQA_image_dir_path", type=str, default="./data/ocrVQA/images")
    parser.add_argument("--ocrVQA_ann_path", type=str, default="./data/ocrVQA/dataset.json")

    #STVQA
    parser.add_argument("--STVQA_image_dir_path", type=str, default="./data/STVQA")
    parser.add_argument("--STVQA_ann_path", type=str, default="./data/STVQA/train_task_3.json")

    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_ocrVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )
    parser.add_argument(
        "--eval_STVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on STVQA."
    )
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )
    #BLIP2
    #parser.add_argument("--BLIP2_model_path", type=str, default="/home/zhangli/GPT4/BLIP2-flant5")
    parser.add_argument("--BLIP2_model_name", type=str, default="blip2_opt")#blip2_t5  blip2_opt blip2_vicuna_instruct
    parser.add_argument("--BLIP2_model_type", type=str, default="pretrain_opt6.7b")#pretrain_flant5xxl pretrain_opt6.7b vicuna13b


    parser.add_argument("--model_name", type=str, default="BLIP2")#mPLUG,miniGPT4,LLaVA
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vision-tower", type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument("--conv-mode", type=str, default="multimodal")
    parser.add_argument("--lora-weight", type=str)

    args = parser.parse_args()
    return args

def main(args):
    np.random.seed(0)
    max_sample_num = 5000
    model, tokenizer  = get_model(args)
    '''ocr_dataset_name=['IIIT5K','svt','IC13_857','IC15_1811','svtp','ct80',
                  'cocotext','ctw','totaltext','HOST','WOST','WordArt']'''
    ocr_dataset_name = args.ocr_dataset_name.split()
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if args.eval_textVQA:
        print("\nEvaluating textVQA")
        dataset = textVQADataset(args.textVQA_image_dir_path, args.textVQA_ann_path)
        # from torch.utils.data import Subset
        # dataset = Subset(dataset, indices=range(10))
        acc = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time, tokenizer=tokenizer)
        result['textVQA'] = acc
    if args.eval_docVQA:
        dataset = docVQADataset(args.docVQA_image_dir_path, args.docVQA_ann_path)
        acc = evaluate_VQA(model, dataset, args.model_name, 'docVQA', time)
        result['docVQA'] = acc

    if args.eval_ocrVQA:
        dataset = ocrVQADataset(args.ocrVQA_image_dir_path, args.ocrVQA_ann_path)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset,random_indices)
        acc = evaluate_VQA(model, dataset, args.model_name, 'ocrVQA', time)
        result['ocrVQA'] = acc
    
    if args.eval_STVQA:
        dataset = STVQADataset(args.STVQA_image_dir_path, args.STVQA_ann_path)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset,random_indices)
        acc = evaluate_VQA(model, dataset, args.model_name, 'STVQA', time)
        result['STVQA'] = acc

    if args.eval_ocr:
        for i in range(len(ocr_dataset_name)):
            dataset = ocrDataset(args.ocr_dir_path, ocr_dataset_name[i])
            acc = evaluate_OCR(model, dataset, args.model_name, ocr_dataset_name[i], time)
            result[ocr_dataset_name[i]] = acc
    result_path = os.path.join(os.path.join(args.answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))
if __name__ == "__main__":
    args = parse_args()
    main(args)