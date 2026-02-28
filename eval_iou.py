import os
import re
import json
import torch
import ast
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = "saves/Qwen3-VL-2B-Counting-Merged"  # Run on the fine-tuned model
HF_DATASET = "foye501/VLM-Counting-dataset"
EVAL_SAMPLES = 50 # Smaller sample size since IoU evaluation is intensive
IOU_THRESHOLD = 0.5 # Threshold for a predicted box to match a ground truth box

# ==========================================
# 2. Bounding Box & IoU Logic
# ==========================================
def extract_boxes(message_content):
    """Extracts bounding boxes from the model output or ground truth.
       Returns a list of [ymin, xmin, ymax, xmax] lists.
    """
    boxes = []
    # Find all occurrences of <box> [...] </box>
    box_strings = re.findall(r'<box>\s*(\[[0-9,\s]+\])\s*</box>', message_content)
    for box_str in box_strings:
        try:
            # Safely evaluate the string representation of the list
            box = ast.literal_eval(box_str)
            if len(box) == 4:
                boxes.append(box)
        except (ValueError, SyntaxError):
            continue
    return boxes

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two boxes [ymin, xmin, ymax, xmax]."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, yB - yA) * max(0, xB - xA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Add a small epsilon to prevent division by zero
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Matches predicted boxes to ground truth boxes and calculates P/R/F1."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0

    matched_gt = set()
    true_positives = 0

    # For each predicted box, find the best matching ground truth box
    for p_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
                
            iou = calculate_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
                
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0
    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return precision, recall, f1

# ==========================================
# 3. Execution
# ==========================================
def main():
    print(f"Loading {MODEL_PATH} for IoU Evaluation...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    print(f"\nDownloading dataset {HF_DATASET} ...")
    dataset = load_dataset(HF_DATASET, split="train")
    dataset = dataset.shuffle(seed=123).select(range(EVAL_SAMPLES))

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    valid_samples = 0

    print(f"\n--- Starting Bounding Box Evaluation ({len(dataset)} images) ---")
    for item in tqdm(dataset):
        image = item["image"].convert("RGB")
        messages = item["messages"]
        
        user_prompt = messages[0]["content"].replace("<image>", "").replace("<|image_pad|>", "").strip()
        gt_boxes = extract_boxes(messages[1]["content"])
        
        if len(gt_boxes) == 0:
            continue # Skip if no ground truth boxes found
            
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
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        pred_boxes = extract_boxes(output_text)
        
        p, r, f1 = evaluate_boxes(pred_boxes, gt_boxes, IOU_THRESHOLD)
        
        total_precision += p
        total_recall += r
        total_f1 += f1
        valid_samples += 1

    print("\n" + "="*50)
    print("--- Bounding Box Localization Results (IoU >= 0.5) ---")
    print("="*50)
    if valid_samples > 0:
        avg_p = total_precision / valid_samples
        avg_r = total_recall / valid_samples
        avg_f1 = total_f1 / valid_samples
        print(f"Evaluated Samples: {valid_samples}")
        print(f"Mean Precision: {avg_p * 100:.2f}% (Are predicted boxes objects?)")
        print(f"Mean Recall:    {avg_r * 100:.2f}% (Did we find all objects?)")
        print(f"Mean F1-Score:  {avg_f1 * 100:.2f}%")
        
        print("\nConclusion:")
        if avg_f1 > 0.8:
            print("Verdict: EXCELLENT. The model is deeply grounded and predicting accurate coordinates.")
        elif avg_f1 > 0.5:
            print("Verdict: GOOD. The model learned to locate objects, though with some spatial noise.")
        else:
            print("Verdict: POOR. The model learned the format but is hallucinating box locations.")
    else:
        print("No valid samples evaluated.")

if __name__ == "__main__":
    main()
