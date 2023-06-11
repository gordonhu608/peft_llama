  #\torchrun --nnodes=1 --nproc_per_node= #WORLD_SIZE=2 
  #torchrun --nnodes=1 --nproc_per_node=1 \
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
  python \
    llava/train/train_insblip_peft.py \
    --model_name_or_path ./data/llama_vicuna_7b\
    --data_path  ./data/textVQA/TextVQA_0.5.1_train.json \
    --image_folder ./data/textVQA/train_images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --tune_mm_mlp_adapter True \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/textocronly_insblip_epoch2_b32\
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --lazy_preprocess True \
    --freeze_qformer False \
    --report_to wandb


    #../pandagpt_visual_instruction_dataset/llava_pandagpt4_visual_instruction_data.json  data/llava_instruct_150k.json 

    #../pandagpt_visual_instruction_dataset/images data/train2017/ 