import os
import re
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# ==========================================
# 1. Configuration
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"  # Base model path
LORA_PATH = "saves/Qwen3-VL-2B-Counting/lora/sft" # Where LLaMA-Factory saved the adapter
MERGED_SAVE_PATH = "saves/Qwen3-VL-2B-Counting-Merged"
HF_DATASET = "foye501/VLM-Counting-dataset"
EVAL_SAMPLES = 200 # How many random images to evaluate to get a representative score

# ==========================================
# 2. Merge LoRA (if not already merged)
# ==========================================
print(f"Loading Base Model: {BASE_MODEL_ID} ...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

if not os.path.exists(MERGED_SAVE_PATH):
    print("Loading Base model into RAM for merging...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="cpu" # Load on CPU to prevent VRAM OOM during merge
    )
    
    print(f"Loading LoRA adapter from {LORA_PATH} ...")
    model_to_merge = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("Merging weights (this transforms the adapter into a standalone model)...")
    merged_model = model_to_merge.merge_and_unload()
    
    print(f"Saving combined model to {MERGED_SAVE_PATH} ...")
    merged_model.save_pretrained(MERGED_SAVE_PATH)
    processor.save_pretrained(MERGED_SAVE_PATH)
    print("Merge successful!")
    
    del base_model
    del model_to_merge
    del merged_model
    torch.cuda.empty_cache()
else:
    print(f"Combined model already detected at {MERGED_SAVE_PATH}!")

# ==========================================
# 3. Evaluation Function
# ==========================================
def extract_ground_truth(assistant_message):
    match = re.search(r'Total count:\s*(\d+)', assistant_message, re.IGNORECASE)
    return int(match.group(1)) if match else -1

def run_evaluation(model_path, dataset_subset, name="Model"):
    print(f"\nLoading {name} for Inference...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    correct = 0
    total_error = 0
    valid_samples = 0
    
    print(f"\n--- Starting {name} Evaluation ({len(dataset_subset)} images) ---")
    for i, item in enumerate(tqdm(dataset_subset)):
        image = item["image"].convert("RGB")
        messages = item["messages"]
        
        # User prompt is message 0, Assistant ground truth is message 1
        user_prompt = messages[0]["content"].replace("<image>", "").replace("<|image_pad|>", "").strip()
        gt_count = extract_ground_truth(messages[1]["content"])
        
        if gt_count == -1:
            continue # Skip invalid training data
            
        # Format for processor
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt} 
                ]
            }
        ]
        
        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        match = re.search(r'Total count:\s*(\d+)', output_text, re.IGNORECASE)
        pred_count = int(match.group(1)) if match else -1
        
        if pred_count != -1:
            valid_samples += 1
            if pred_count == gt_count:
                correct += 1
            total_error += abs(pred_count - gt_count)
            
    print(f"\n--- {name} Results ---")
    if valid_samples > 0:
        print(f"Accuracy: {(correct/valid_samples)*100:.1f}% ({correct}/{valid_samples})")
        print(f"Mean Absolute Error (MAE): {total_error/valid_samples:.2f}")
    else:
        print("No valid samples evaluated.")
        
    del model
    torch.cuda.empty_cache()

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"\nDownloading dataset {HF_DATASET} ...")
    dataset = load_dataset(HF_DATASET, split="train")
    
    # Shuffle and pick N samples to evaluate so it doesn't take 10 hours
    print(f"Shuffling dataset and selecting {EVAL_SAMPLES} samples...")
    dataset = dataset.shuffle(seed=42).select(range(EVAL_SAMPLES))
    
    # 1. Evaluate Base Model
    run_evaluation(BASE_MODEL_ID, dataset, name="Base Model (Qwen3-VL-2B)")
    
    # 2. Evaluate Fine-Tuned (Merged) Model
    run_evaluation(MERGED_SAVE_PATH, dataset, name="Fine-Tuned Model (Merged)")
    
    print("\nEvaluation Complete!")
