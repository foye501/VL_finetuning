import os
import re
import json
import torch
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def save_annotated_image(image, gt_boxes, pred_boxes, output_path, iou_f1):
    """Draws GT (green) and Pred (red) boxes on the image and saves it."""
    # Convert PIL Image to physical dimensions for Matplotlib
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Width and height of the image to denormalize 0-1000 coordinates
    width, height = image.size
    
    # Helper to convert [ymin, xmin, ymax, xmax] from 0-1000 scale to pixel coordinates
    def convert_box(box):
        y1, x1, y2, x2 = box
        px1 = (x1 / 1000.0) * width
        py1 = (y1 / 1000.0) * height
        px2 = (x2 / 1000.0) * width
        py2 = (y2 / 1000.0) * height
        return px1, py1, px2 - px1, py2 - py1

    # Draw Ground Truth boxes (Green)
    for box in gt_boxes:
        x, y, w, h = convert_box(box)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none', linestyle='dashed')
        ax.add_patch(rect)

    # Draw Predicted boxes (Red)
    for box in pred_boxes:
        x, y, w, h = convert_box(box)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Ground Truth (Green): {len(gt_boxes)} | Predicted (Red): {len(pred_boxes)} | F1: {iou_f1:.2f}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def get_difficulty(count):
    if count <= 5: return "Easy"
    if count <= 20: return "Medium"
    if count <= 50: return "Hard"
    return "Extreme"

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

    tier_stats = {
        "Easy": {"p_sum": 0, "r_sum": 0, "f1_sum": 0, "count": 0},
        "Medium": {"p_sum": 0, "r_sum": 0, "f1_sum": 0, "count": 0},
        "Hard": {"p_sum": 0, "r_sum": 0, "f1_sum": 0, "count": 0},
        "Extreme": {"p_sum": 0, "r_sum": 0, "f1_sum": 0, "count": 0}
    }
    
    vis_dir = "eval_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    saved_visualizations = 0

    print(f"\n--- Starting Bounding Box Evaluation ({len(dataset)} images) ---")
    for item in tqdm(dataset):
        image = item["image"].convert("RGB")
        messages = item["messages"]
        
        user_prompt = messages[0]["content"].replace("<image>", "").replace("<|image_pad|>", "").strip()
        gt_boxes = extract_boxes(messages[1]["content"])
        
        target_count = len(gt_boxes)
        if target_count == 0:
            continue # Skip if no ground truth boxes found
            
        tier = get_difficulty(target_count)
            
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
        
        # Save visualizations for up to 5 Hard samples
        if tier == "Hard" and saved_visualizations < 5:
            vis_path = os.path.join(vis_dir, f"hard_sample_{saved_visualizations+1}.png")
            save_annotated_image(image, gt_boxes, pred_boxes, vis_path, f1)
            saved_visualizations += 1
        
        tier_stats[tier]["p_sum"] += p
        tier_stats[tier]["r_sum"] += r
        tier_stats[tier]["f1_sum"] += f1
        tier_stats[tier]["count"] += 1

    print("\n" + "="*50)
    print("--- Bounding Box Localization Results (IoU >= 0.5) ---")
    print("="*50)
    
    total_valid = sum(t["count"] for t in tier_stats.values())
    
    if total_valid > 0:
        overall_p = sum(t["p_sum"] for t in tier_stats.values()) / total_valid
        overall_r = sum(t["r_sum"] for t in tier_stats.values()) / total_valid
        overall_f1 = sum(t["f1_sum"] for t in tier_stats.values()) / total_valid
        
        print(f"Overall Evaluated Samples: {total_valid}")
        print(f"Overall Mean Precision: {overall_p * 100:.2f}% (Are predicted boxes objects?)")
        print(f"Overall Mean Recall:    {overall_r * 100:.2f}% (Did we find all objects?)")
        print(f"Overall Mean F1-Score:  {overall_f1 * 100:.2f}%")
        
        print("\nBreakdown by Difficulty Tier:")
        for tier_name, stats in tier_stats.items():
            if stats["count"] > 0:
                tier_p = stats["p_sum"] / stats["count"]
                tier_r = stats["r_sum"] / stats["count"]
                tier_f1 = stats["f1_sum"] / stats["count"]
                print(f"  {tier_name} (n={stats['count']}):")
                print(f"    Precision: {tier_p*100:.2f}%")
                print(f"    Recall:    {tier_r*100:.2f}%")
                print(f"    F1-Score:  {tier_f1*100:.2f}%")
        
        print("\nConclusion:")
        if overall_f1 > 0.8:
            print("Verdict: EXCELLENT. The model is deeply grounded and predicting accurate coordinates.")
        elif overall_f1 > 0.5:
            print("Verdict: GOOD. The model learned to locate objects, though with some spatial noise.")
        else:
            print("Verdict: POOR. The model learned the format but is hallucinating box locations.")
    else:
        print("No valid samples evaluated.")

if __name__ == "__main__":
    main()
