import torch 

if __name__ == "__main__":
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
    state_dict = torch.load('checkpoints/pretrain_blip_projection_with_text/pytorch_model-00003-of-00003.bin')
    for key in state_dict.keys():
        if key.startswith("model.Qformer.bert.encoder.layer.5."):
            print(key, state_dict[key])
        #if torch.all(torch.isnan(state_dict[key])):
            #print("nan found", key)
    #print("state_dict", state_dict.keys())