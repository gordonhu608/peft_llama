WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nnodes=1 --nproc_per_node=2 --master_port=25001 \
    llava/train/peft_train.py \
    --model_name_or_path data/LLaVA-7B-v0 \
    --data_path data/llava_instruct_150k.json  \
    --image_folder data/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/reproduce_stage_2_findout \
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
    --ddp_find_unused_parameters False \
    --lazy_preprocess True \
    --report_to wandb