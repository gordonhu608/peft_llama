#!/bin/bash
model_name="./checkpoints/pretrain_blip_projection_with_text/"
lora_weight="./save_pretrained/stage2_fully_blip_epoch1"
answers_file="data/OwlEval/answer/minigpt4_13b_answer.jsonl"

# Get the last part of the path
output_file="${answers_file##*/}"

# Concatenate string in bash
output_name="owl_vs_${output_file}"

echo "export OPENAI_API_KEY=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" >> ~/.zshrc
source ~/.zshrc

python llava/eval/evaluation_judged_by_gpt.py \
-a "data/OwlEval/answer/mPLUG_Owl_7b_answer.jsonl" $answers_file \
-o "outputs/gpt_eval_compare/$output_name"

python llava/eval/summarize_gpt_review.py  \
-a "outputs/gpt_eval_compare/$output_name"







