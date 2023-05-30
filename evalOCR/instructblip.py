if __name__ == "__main__":    

    import torch
    from lavis.models import load_model_and_preprocess

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct", 
        model_type="vicuna7b", 
        is_eval=True, 
        device=device
    )

