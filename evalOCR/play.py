
if __name__ == "__main__":
    
    from llava.model.dist_utils import download_cached_file
    import torch 
    import os 


    instruct_qformer = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth"
    cached_file = download_cached_file(
                instruct_qformer, check_hash=False, progress=True
            )
    q_former_checkpoint = torch.load(cached_file, map_location="cpu")
    q_former_state_dict = q_former_checkpoint["model"]
    q_former_opt_state_dict = q_former_checkpoint["optimizer"]
    q_former_config = q_former_checkpoint["config"]
    q_former_scaler = q_former_checkpoint["scaler"]
    q_former_iters = q_former_checkpoint["iters"]
    print(q_former_state_dict.keys())