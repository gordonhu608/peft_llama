import torch 
import sys
from pdb import set_trace 

if __name__ == "__main__":
    from llava.model.dist_utils import download_cached_file


    state = torch.load('checkpoints/trained_mixed_insblip_epoch1/pytorch_model.bin')
    new_state = {'.'.join(k.split('.')[2:]): v for k,v in state.items()}
    print(new_state.keys())
    
    torch.save(new_state, 'checkpoints/trained_mixed_insblip_epoch1/pytorch_model.bin' )
    
    sys.exit(1)

    q_former_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt6.7b.pth"
    #"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
    #"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
    #"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"
    cached_file = download_cached_file(
            q_former_model, check_hash=False, progress=True
        )
    q_former_checkpoint = torch.load(cached_file, map_location="cpu") 
    q_former_state_dict = q_former_checkpoint["model"]
    print(q_former_state_dict.keys())
    set_trace()
    
    del q_former_state_dict['ln_vision.weight']
    del q_former_state_dict['ln_vision.bias']
    # del q_former_state_dict['llm_proj.weight']
    # del q_former_state_dict['llm_proj.bias']
    #del q_former_state_dict['text_proj.weight']
    #del q_former_state_dict['text_proj.bias']
    del q_former_state_dict['query_tokens']


    ins_model = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"
    #"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
    #
    cached_file = download_cached_file(
            ins_model, check_hash=False, progress=True
        )
    ins_checkpoint = torch.load(cached_file, map_location="cpu") 
    ins_state_dict = ins_checkpoint["model"]

    del ins_state_dict['ln_vision.weight']
    del ins_state_dict['ln_vision.bias']
    del ins_state_dict['llm_proj.weight']  # 4096 768
    del ins_state_dict['llm_proj.bias']
    del ins_state_dict['query_tokens']

    for k,v in q_former_state_dict.items():
        if k not in ins_state_dict:
            print("not in ins_state_dict", k)
    
    #vision_proj 256, 768

    from pdb import set_trace 
    set_trace()


    q_former_state_dict["vision_proj.weight"].shape
    '''
    state_dict_1 = torch.load('data/vicuna_7b/pytorch_model-00001-of-00002.bin')
    state_dict_2 = torch.load('data/vicuna_7b/pytorch_model-00002-of-00002.bin')

    # Merge the state dictionaries
    state_dict = {**state_dict_1, **state_dict_2}
    #print("state_dict", state_dict.keys())
    # Modify the weights
    for name in ['model.embed_tokens.weight', 'lm_head.weight']:
        old_weight = state_dict[name].data
        print("old_weight", old_weight.shape)
        average_weight = old_weight.mean(dim=0, keepdim=True)
        new_weight = torch.cat([old_weight, average_weight.repeat(4, 1)], dim=0)
        print("new_weight", new_weight.shape)
        state_dict[name] = new_weight

    # Save the modified state dictionary
    torch.save(state_dict, 'data/modified_pytorch_model.bin')
    '''

    
    
    state_dict_stage2 = torch.load("./save_pretrained/stage2_only_llm_blip_instruct/adapter_model.bin") #("checkpoints/stage2_instruct_blip/pytorch_model.bin") #('checkpoints/pretrain_blip_projection_with_text/pytorch_model-00003-of-00003.bin')
    state_dict_stage1 = torch.load("checkpoints/pretrain_blip_projection_with_text/pytorch_model-00003-of-00003.bin")
    check_keys = ["model.Qformer.bert.encoder.layer.1.output_query.dense.weight", "model.Qformer.bert.encoder.layer.0.attention.output.LayerNorm.bias",
                    "model.Qformer.bert.encoder.layer.0.attention.output.LayerNorm.weight",
                    "model.Qformer.bert.encoder.layer.0.attention.output.dense.bias",
                    "model.Qformer.bert.encoder.layer.0.attention.output.dense.weight",
                    "model.Qformer.bert.encoder.layer.0.attention.self.key.bias",
                    "model.Qformer.bert.encoder.layer.0.attention.self.key.weight",
                    "model.Qformer.bert.encoder.layer.0.attention.self.query.bias",
                    "model.Qformer.bert.encoder.layer.0.attention.self.query.weight",
                    "model.Qformer.bert.encoder.layer.0.attention.self.value.bias",
                    "model.Qformer.bert.encoder.layer.0.attention.self.value.weight"]
    check_key_2 = ["base_model.model." + check_key for check_key in check_keys]
    for i,j in zip(check_keys, check_key_2):
        print(torch.equal(state_dict_stage1[i], state_dict_stage2[j].cpu()))
    # for key in state_dict_stage2.keys():
    #     print(key)
        # if key.startswith("model.Qformer.bert.encoder.layer.5."):
        #     print(key, state_dict[key])
        #if torch.all(torch.isnan(state_dict[key])):
            #print("nan found", key)
    #print("state_dict", state_dict.keys())