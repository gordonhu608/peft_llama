


python eval_onestage_peft.py \
    --model_name ../checkpoints/textocronly_insblip_epoch2_b32/ \
    --lora-weight ../save_pretrained/trained_mixed_insblip_epoch1/ \
    --device cuda:0 \
    --eval_textVQA