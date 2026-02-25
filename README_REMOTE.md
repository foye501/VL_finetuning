# Remote Fine-Tuning Guide (CUDA Server)

Follow these exact steps on your **remote GPU server** to download your dataset and fine-tune Qwen2-VL using LLaMA-Factory.

## Step 1: Clone Your Repository & Install LLaMA-Factory

Run this in your remote server's terminal:

```bash
# 1. Clone your project code
git clone https://github.com/foye501/VL_finetuning.git
cd VL_finetuning

# 2. Clone and install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Step 2: Authenticate with Hugging Face

Since your dataset (`foye501/VLM-Counting-dataset`) is private, your remote server needs permission to download it.

```bash
# Log in to Hugging Face
huggingface-cli login
# Paste your Access Token from https://huggingface.co/settings/tokens
```

## Step 3: Register the Dataset in LLaMA-Factory

LLaMA-Factory can stream the dataset directly from Hugging Face! You just need to tell it where to look.

Open `LLaMA-Factory/data/dataset_info.json` on your remote server, and add this block inside the main JSON oibject:

```json
  "vlm_counting": {
    "hf_hub_url": "foye501/VLM-Counting-dataset",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "image"
    }
  },
```

## Step 4: Launch the Fine-Tuning Job

Finally, run the LLaMA-Factory training command targeting `v_proj` (Vision Projector) and `q_proj` (Attention Decoder). 

Run this command while inside the `LLaMA-Factory` directory:

```bash
export CUDA_VISIBLE_DEVICES=0

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset vlm_counting \
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
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --val_size 0.05 \
    --load_best_model_at_end \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 true
```

*Note: The `--dataset vlm_counting` argument tells LLaMA-Factory to automatically download your 10,000 images from Hugging Face based on the `hf_hub_url` you added in Step 3.*
