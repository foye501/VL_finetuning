import os
import re
import math
import torch
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = "saves/Qwen3-VL-2B-Counting-Merged"  # Must use the fine-tuned model
HF_DATASET = "foye501/VLM-Counting-dataset"
EVAL_SAMPLES = 20 # Small sample to test the pipeline
NMS_IOU_THRESHOLD = 0.3 # Threshold to merge overlapping boxes on the seams
GRID_SIZE = 2 # 2x2 grid (4 patches)

# ==========================================
# 2. Bounding Box Math (NMS & Transformation)
# ==========================================
def extract_boxes(message_content):
    """Parses <box> [ymin, xmin, ymax, xmax] </box> from text."""
    boxes = []
    box_strings = re.findall(r'<box>\s*(\[[0-9,\s]+\])\s*</box>', message_content)
    for box_str in box_strings:
        try:
            box = ast.literal_eval(box_str)
            if len(box) == 4:
                boxes.append(box)
        except:
            continue
    return boxes

def calculate_iou(boxA, boxB):
    """Calculates traditional IoU for NMS."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, yB - yA) * max(0, xB - xA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def apply_nms(boxes, iou_threshold=0.3):
    """Applies Non-Maximum Suppression to remove duplicates along the grid seams.
       (Since we don't have object confidence scores natively, we just merge eagerly).
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by area (largest to smallest) as a proxy for confidence, 
    # to keep the most encompassing box if they overlap heavily on a seam
    areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
    sorted_indices = np.argsort(areas)[::-1]
    sorted_boxes = [boxes[i] for i in sorted_indices]
    
    keep = []
    while len(sorted_boxes) > 0:
        # Pick the box with largest area
        curr_box = sorted_boxes.pop(0)
        keep.append(curr_box)
        
        # Compare to all remaining boxes
        remaining = []
        for next_box in sorted_boxes:
            if calculate_iou(curr_box, next_box) < iou_threshold:
                remaining.append(next_box)
        sorted_boxes = remaining
        
    return keep

# ==========================================
# 3. Agentic Slicing Flow
# ==========================================
def process_high_res_image(image, prompt_text, model, processor, grid_dim=2):
    """
    Slices the image into a grid_dim x grid_dim matrix.
    Passes each slice to the VLM independently.
    Translates the output coordinates back to the global 0-1000 scale.
    """
    width, height = image.size
    tile_w = width // grid_dim
    tile_h = height // grid_dim
    
    global_boxes = []
    
    # 1. Slice and process 
    for row in range(grid_dim):
        for col in range(grid_dim):
            # Calculate physical pixel boundaries for this tile
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            
            # Crop the tile
            tile_img = image.crop((left, upper, right, lower))
            
            # CRITICAL FIX: Qwen-VL uses adaptive token scaling based on input dimension.
            # If we just pass a smaller crop, it allocates fewer vision tokens, defeating the purpose!
            # We must upsample the crop back to the original dimensions to force maximum token allocation.
            tile_img = tile_img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Prepare VLM Input
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": tile_img},
                        {"type": "text", "text": prompt_text} 
                    ]
                }
            ]
            text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[tile_img], padding=True, return_tensors="pt").to(model.device)

            # Inference
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            # Extract local boxes (these are 0-1000 relative to the TILE, not the whole image)
            local_boxes = extract_boxes(output_text)
            
            # 2. Coordinate Translation
            # We must map the 0-1000 local scale back to the 0-1000 global scale
            for l_box in local_boxes:
                # l_box format: [ymin, xmin, ymax, xmax] 
                l_ymin, l_xmin, l_ymax, l_xmax = l_box
                
                # Convert back to physical pixels within the tile
                px_ymin = (l_ymin / 1000.0) * tile_h
                px_xmin = (l_xmin / 1000.0) * tile_w
                px_ymax = (l_ymax / 1000.0) * tile_h
                px_xmax = (l_xmax / 1000.0) * tile_w
                
                # Add the absolute pixel offset for this tile
                global_px_ymin = px_ymin + upper
                global_px_xmin = px_xmin + left
                global_px_ymax = px_ymax + upper
                global_px_xmax = px_xmax + left
                
                # Convert back to 0-1000 global scale
                g_ymin = int((global_px_ymin / height) * 1000)
                g_xmin = int((global_px_xmin / width) * 1000)
                g_ymax = int((global_px_ymax / height) * 1000)
                g_xmax = int((global_px_xmax / width) * 1000)
                
                # Clamp between 0-1000
                global_boxes.append([
                    max(0, min(1000, g_ymin)),
                    max(0, min(1000, g_xmin)),
                    max(0, min(1000, g_ymax)),
                    max(0, min(1000, g_xmax))
                ])

    # 3. Apply NMS to remove boxes duplicated across the grid cut-lines
    final_boxes = apply_nms(global_boxes, iou_threshold=NMS_IOU_THRESHOLD)
    return final_boxes

def save_comparison_image(image, gt_boxes, single_pass_boxes, tiled_boxes, output_path):
    """Draws GT (Green), Single Pass Error (Red), and Tiled Solution (Blue)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    width, height = image.size
    
    def draw_boxes(ax, title, boxes, color, linestyle='-'):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
        
        # Draw GT on both for reference
        for box in gt_boxes:
            y1, x1, y2, x2 = box
            px1, py1 = (x1/1000)*width, (y1/1000)*height
            w, h = (x2-x1)/1000*width, (y2-y1)/1000*height
            rect = patches.Rectangle((px1, py1), w, h, linewidth=2, edgecolor='g', facecolor='none', linestyle='dashed')
            ax.add_patch(rect)
            
        # Draw the target boxes
        for box in boxes:
            y1, x1, y2, x2 = box
            px1, py1 = (x1/1000)*width, (y1/1000)*height
            w, h = (x2-x1)/1000*width, (y2-y1)/1000*height
            rect = patches.Rectangle((px1, py1), w, h, linewidth=2, edgecolor=color, facecolor='none', linestyle=linestyle)
            ax.add_patch(rect)

    draw_boxes(axes[0], f"Standard Base Failure (Red)\nGT Count: {len(gt_boxes)} | Pred: {len(single_pass_boxes)}", single_pass_boxes, 'r')
    draw_boxes(axes[1], f"Track 3 Tiled High-Res + NMS (Blue)\nGT Count: {len(gt_boxes)} | Pred: {len(tiled_boxes)}", tiled_boxes, 'b')
    
    # Draw cut lines to visualize the patching on the Blue image
    tile_w = width // GRID_SIZE
    tile_h = height // GRID_SIZE
    for row in range(1, GRID_SIZE):
        axes[1].axhline(y=row*tile_h, color='cyan', linestyle=':', linewidth=2)
    for col in range(1, GRID_SIZE):
        axes[1].axvline(x=col*tile_w, color='cyan', linestyle=':', linewidth=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# ==========================================
# 4. Main
# ==========================================
def main():
    print(f"Loading {MODEL_PATH} for Track 3 Evaluation...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    dataset = load_dataset(HF_DATASET, split="train").shuffle(seed=42).select(range(200)) # select from large enough pool to find dense images
    
    vis_dir = "track3_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    images_saved = 0

    print("Hunting for Hard/Extreme difficulty images (count > 25)...")
    
    for item in tqdm(dataset):
        if images_saved >= EVAL_SAMPLES:
            break
            
        image = item["image"].convert("RGB")
        messages = item["messages"]
        user_prompt = messages[0]["content"].replace("<image>", "").replace("<|image_pad|>", "").strip()
        gt_boxes = extract_boxes(messages[1]["content"])
        
        if len(gt_boxes) < 25:
            continue # We only care about fixing the hard/extreme ones now!
            
        # 1. Base Single Pass 
        chat = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
        ]
        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=1024)
            
        gen_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
        single_pass_boxes = extract_boxes(processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0])
        
        # 2. Track 3: Tiled Processing
        tiled_boxes = process_high_res_image(image, user_prompt, model, processor, grid_dim=GRID_SIZE)
        
        # Save visualization comparing them
        vis_path = os.path.join(vis_dir, f"track3_compare_{images_saved+1}.png")
        save_comparison_image(image, gt_boxes, single_pass_boxes, tiled_boxes, vis_path)
        
        images_saved += 1
        
    print(f"\nDone! Saved {images_saved} visualization comparisons to {vis_dir}/")

if __name__ == "__main__":
    main()
