import os
import random
import json
import math
from PIL import Image, ImageDraw
import uuid

import config

def ensure_dirs():
    """Ensure data directories exist."""
    os.makedirs(config.IMAGE_DIR, exist_ok=True)

def random_point(margin, size):
    """Generate a random (x, y) point within the image, avoiding margins."""
    w, h = config.IMAGE_SIZE
    return (
        random.randint(margin, w - margin - size * 2),
        random.randint(margin, h - margin - size * 2)
    )

def check_overlap(box1, box2):
    """Check if two bounding boxes (x1, y1, x2, y2) overlap."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # If one rectangle is on left side of other
    if x1_max < x2_min or x2_max < x1_min:
        return False
    # If one rectangle is above other
    if y1_max < y2_min or y2_max < y1_min:
        return False
    return True

def draw_shape(draw, shape_type, color, x, y, size):
    """Draw a specific shape using Pillow."""
    bbox = (x, y, x + size * 2, y + size * 2)
    
    if shape_type == "circle":
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "square":
        draw.rectangle(bbox, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "triangle":
        # Pointing up
        points = [
            (x + size, y),
            (x, y + size * 2),
            (x + size * 2, y + size * 2)
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "star":
        # 5-pointed star approx
        cx, cy = x + size, y + size
        r_outer = size
        r_inner = size * 0.4
        points = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            r = r_outer if i % 2 == 0 else r_inner
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "diamond":
        points = [
            (x + size, y),
            (x, y + size),
            (x + size, y + size * 2),
            (x + size * 2, y + size)
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "pentagon":
        cx, cy = x + size, y + size
        r = size
        points = []
        for i in range(5):
            angle = i * 2 * math.pi / 5 - math.pi / 2
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape_type == "hexagon":
        cx, cy = x + size, y + size
        r = size
        points = []
        for i in range(6):
            angle = i * 2 * math.pi / 6 - math.pi / 2
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
        
    return bbox

def generate_image(difficulty_level, num_samples, start_idx=0):
    """Generate images and annotations for a specific difficulty level."""
    params = config.DIFFICULTY_LEVELS[difficulty_level]
    annotations = []
    
    for i in range(num_samples):
        # 1. Setup scene parameters
        target_count = random.randint(*params["count_range"])
        target_shape = random.choice(config.SHAPES)
        target_color = random.choice(config.TARGET_COLORS)
        
        # Color string mapping for the question
        color_idx = config.TARGET_COLORS.index(target_color)
        color_names = ["red", "blue", "green", "yellow", "purple", "orange"]
        target_color_name = color_names[color_idx]
        
        bg_name = random.choice(params["backgrounds"])
        
        # Handle complex backgrounds
        if bg_name in ["gradient", "noisy", "textured", "complex"]:
            # Simplify for MVP: just pick a random solid background color
            # Future enhancement: actual noisy/gradient generation
            bg_color = (
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255)
            )
        else:
            bg_color = config.BACKGROUND_COLORS[bg_name]
            
        img = Image.new("RGB", config.IMAGE_SIZE, bg_color)
        draw = ImageDraw.Draw(img)
        
        # 2. Setup distractors
        num_distractors = 0
        distractor_shapes = []
        
        if params["max_distractors"] > 0:
            num_distractors = random.randint(0, min(params["max_distractors"], target_count * 2))
            
            # Choose distractor properties
            available_shapes = [s for s in config.SHAPES if s != target_shape]
            for _ in range(num_distractors):
                d_shape = random.choice(available_shapes)
                d_color = random.choice(config.DISTRACTOR_COLORS)
                distractor_shapes.append((d_shape, d_color))
                
        # 3. Place objects safely
        placed_boxes = []
        target_boxes = []
        all_objects_to_place = [(target_shape, target_color)] * target_count + distractor_shapes
        random.shuffle(all_objects_to_place)
        
        size_min, size_max = config.OBJECT_SIZE_RANGE[difficulty_level]
        actual_distractors_placed = 0
        actual_targets_placed = 0
        
        has_overlap = False
        
        for shape, color in all_objects_to_place:
            size = random.randint(size_min, size_max)
            placed = False
            attempts = 0
            
            while not placed and attempts < 100:
                x, y = random_point(margin=10, size=size)
                new_box = (x, y, x + size * 2, y + size * 2)
                
                # Check overlap if not allowed
                conflict = False
                if not params["allow_overlap"]:
                    for box in placed_boxes:
                        if check_overlap(new_box, box):
                            conflict = True
                            break
                elif params["allow_overlap"] and attempts > 50:
                    # Allow overlap after 50 failed non-overlap attempts to pack dense scenes
                    conflict = False
                    has_overlap = True
                else:
                    for box in placed_boxes:
                        if check_overlap(new_box, box):
                            conflict = True
                            has_overlap = True
                            break
                            
                if not conflict:
                    placed_box = draw_shape(draw, shape, color, x, y, size)
                    placed_boxes.append(placed_box)
                    if shape == target_shape and color == target_color:
                        target_boxes.append(placed_box)
                        actual_targets_placed += 1
                    else:
                        actual_distractors_placed += 1
                    placed = True
                
                attempts += 1
                
            # If we absolutely cannot place it (density too high), break out to avoid infinite loops
            # But we must update the target_count to reflect physical reality!
            if not placed and shape == target_shape and color == target_color:
                # We couldn't place a target, log it but don't count it
                pass
                
        # Note: we MUST use actual_targets_placed as the ground truth answer,
        # in case packing failed for very large counts
        
        # 4. Save Image
        img_id = f"{difficulty_level}_{i+start_idx:05d}_{uuid.uuid4().hex[:6]}"
        img_filename = f"{img_id}.png"
        img_path = os.path.join(config.IMAGE_DIR, img_filename)
        img.save(img_path)
        
        # 5. Generate Question
        if actual_distractors_placed > 0:
            template = random.choice(config.QUESTION_TEMPLATES)
            q = template.format(color=target_color_name, shape=target_shape)
        else:
            # If no distractors, we can ask generic "how many squares" without color specify
            template = random.choice(config.QUESTION_TEMPLATES_NO_COLOR + config.QUESTION_TEMPLATES)
            q = template.format(color=target_color_name, shape=target_shape)
            
        # 6. Record Annotation for standard evaluation
        base_annotation = {
            "id": img_id,
            "image_path": f"{config.IMAGE_DIR}/{img_filename}",
            "question": q,
            "answer": actual_targets_placed,
            "difficulty": difficulty_level,
            "target_shape": target_shape,
            "target_color": target_color_name,
            "num_distractors": actual_distractors_placed,
            "has_overlap": has_overlap,
        }
        
        # 7. Record Annotation for LLaMA-Factory Fine-tuning
        # Qwen-VL expects boxes in format: <box> [ymin, xmin, ymax, xmax] </box> relative to 1000
        w, h = config.IMAGE_SIZE
        box_sequence = ""
        for box in target_boxes:
            xmin, ymin, xmax, ymax = box
            # Normalize to 0-1000 range and clamp to ensure vocab safety
            nx_min = max(0, min(1000, int((xmin / w) * 1000)))
            ny_min = max(0, min(1000, int((ymin / h) * 1000)))
            nx_max = max(0, min(1000, int((xmax / w) * 1000)))
            ny_max = max(0, min(1000, int((ymax / h) * 1000)))
            box_sequence += f"<box> [{ny_min}, {nx_min}, {ny_max}, {nx_max}] </box> "
            
        assistant_response = f"{box_sequence.strip()}\nTotal count: {actual_targets_placed}".strip()
        
        llama_factory_item = {
            "messages": [
                {
                    "content": "<image>" + q + " Please annotate the location of each object, and then state the total count.",
                    "role": "user"
                },
                {
                    "content": assistant_response,
                    "role": "assistant"
                }
            ],
            "images": [
                f"{config.IMAGE_DIR}/{img_filename}"
            ]
        }

        annotations.append({
            "base": base_annotation,
            "llama_factory": llama_factory_item
        })
        
    return annotations

def main():
    print("Setting up VLM counting dataset generator...")
    random.seed(config.RANDOM_SEED)
    ensure_dirs()
    
    # Scale up for fine-tuning (e.g. 10k total images)
    # We will grab 2.5k from each difficulty tier
    num_train_samples = 2500
    
    all_llama_factory_data = []
    all_eval_data = []
    
    for level, params in config.DIFFICULTY_LEVELS.items():
        print(f"Generating {num_train_samples} samples for '{level}' difficulty...")
        level_annotations = generate_image(level, num_train_samples)
        
        for item in level_annotations:
            all_eval_data.append(item["base"])
            all_llama_factory_data.append(item["llama_factory"])
            
    # Save standard JSON for internal eval
    with open(config.ANNOTATION_FILE, "w") as f:
        json.dump(all_eval_data, f, indent=2)
        
    # Save LLaMA-Factory JSON for fine-tuning
    llama_factory_file = os.path.join(config.DATA_DIR, "vlm_counting_train.json")
    with open(llama_factory_file, "w") as f:
        json.dump(all_llama_factory_data, f, indent=2)
        
    print(f"\nDone! Generated {len(all_eval_data)} total images.")
    print(f"Eval Annotations saved to: {config.ANNOTATION_FILE}")
    print(f"LLaMA-Factory dataset saved to: {llama_factory_file}")

if __name__ == "__main__":
    main()