#!/bin/bash
# read -p "Enter model backbone name [default: llama]: " backbone
backbone=${backbone:-"Llama-3.2-3B-Instruct"} # #Llama-3.2-3B-Instruct  Qwen2.5-7B-Instruct

export CUDA_VISIBLE_DEVICES=2
data_name="rank_zephyr_IR_20" #("rank_zephyr_IR_20" "quora_25" "ml-1m_25" "Games_25") 
variant="wo_HAS"
n_group=5
echo "Training with data_name: $data_name, variant: $variant, n_group: $n_group"

python Mobius_inversion.py \
    --backbone "$backbone" \
    --data_name "$data_name" \
    --model_path "/data/zoo/${backbone}" \
    --learning_rate 1e-3 \
    --variant $variant \
    --n_group $n_group    
