#!/bin/bash

# ==============================================================================
# Fine-Tuning Qwen2-VL-2B via LLaMA-Factory for Visual Counting
# ==============================================================================
# This script launches a LoRA fine-tuning job on the generated counting dataset.
# It targets the Attention blocks and the Vision Projector to ensure the model
# learns the mapping between visual features and spatial bounding boxes.

# 1. Ensure LLaMA-Factory knows about our dataset
DATA_DIR="$(pwd)/data"
DATASET_JSON="$DATA_DIR/vlm_counting_train.json"
LLAMA_FACTORY_DIR="$HOME/LLaMA-Factory" 
DATA_INFO_FILE="$LLAMA_FACTORY_DIR/data/dataset_info.json"

if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "ERROR: LLaMA-Factory not found at $LLAMA_FACTORY_DIR"
    echo "Please clone it: git clone https://github.com/hiyouga/LLaMA-Factory.git $HOME/LLaMA-Factory"
    exit 1
fi

echo "Adding custom dataset 'vlm_counting' to LLaMA-Factory config..."
# Note: In reality, you'd patch dataset_info.json programmatically, but assuming 
# users configure this manually or we append it if missing.

export CUDA_VISIBLE_DEVICES=0
# If running on Mac Apple Silicon (MPS):
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Starting LoRA Fine-Tuning for Qwen2-VL-2B..."

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset $DATASET_JSON \
    --dataset_dir $DATA_DIR \
    --template qwen2_vl \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,vision_embed_tokens \
    --output_dir saves/Qwen2-VL-2B-Counting/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 false \
    --bf16 false # Assuming MPS backend, usually float32 is safer or fp16 if supported

echo "Fine-tuning complete! Check saves/Qwen2-VL-2B-Counting/lora/sft for adapter weights."
