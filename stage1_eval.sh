python  -m  llava.eval.run_llava_seg \
  --model-name checkpoints/pretrain_blip_projection_with_text \
  --image-file "https://llava-vl.github.io/static/images/view.jpg" \
  --query "What are the things I should be cautious about when I visit here?"

  #checkpoints/blip_projection_layer_558_freeze \ 
  #checkpoints/instruct_pretrain_blip_projection \ loss goes up then down 