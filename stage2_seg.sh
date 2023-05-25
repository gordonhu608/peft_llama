WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
    llava/train/peftacc_train_llama_seg.py \
    --model_name_or_path checkpoints/seg_projection \
    --data_path data/small_llava_instruct_150k.json  \
    --image_folder data/coco_instruct_small \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/stage2_findout \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --lazy_preprocess True \
    --report_to wandb
    #stage2_only_llm_instruct_blip \