import os
import json
import base64
import argparse
from tqdm import tqdm
import google.genai as genai
from google.genai import types
from openai import OpenAI
import config

def file_to_generative_part(path, mime_type):
    """Convert a local file into a generative part format for Gemini APIs."""
    with open(path, "rb") as f:
        data = f.read()
    return types.Part.from_bytes(data=data, mime_type=mime_type)

def encode_image_base64(image_path):
    """Encode an image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def evaluate_model(model_name="gemini-2.5-flash", max_per_level=None, use_vcot=False):
    """Evaluate the generated dataset using a specified VLM."""
    
    # Provider routing
    is_openai = model_name.startswith("gpt")
    
    if is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("Please run: export OPENAI_API_KEY='your_key'")
            return
        client = OpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.")
            print("Please run: export GEMINI_API_KEY='your_key'")
            return
        client = genai.Client(api_key=api_key)
    
    with open(config.ANNOTATION_FILE, "r") as f:
        all_annotations = json.load(f)
        
    # Apply subset limits if requested
    annotations = []
    if max_per_level:
        print(f"Limiting to max {max_per_level} images per difficulty level.")
        counts_by_level = {k: 0 for k in config.DIFFICULTY_LEVELS.keys()}
        for item in all_annotations:
            diff = item["difficulty"]
            if counts_by_level[diff] < max_per_level:
                annotations.append(item)
                counts_by_level[diff] += 1
    else:
        annotations = all_annotations
        
    # Create results directory
    results_dir = os.path.join(config.DATA_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "_vcot" if use_vcot else ""
    results_file = os.path.join(results_dir, f"{model_name}{suffix}_results.json")
    
    results = []
    correct_count = 0
    total_error = 0
    
    print(f"Starting evaluation of {len(annotations)} images using {model_name}...")
    if use_vcot:
        print("Using Visual Chain-of-Thought (vCoT) coordinate tracking!")
    
    difficulty_stats = {diff: {"total": 0, "correct": 0, "abs_error_sum": 0} 
                        for diff in config.DIFFICULTY_LEVELS.keys()}

    for item in tqdm(annotations):
        img_path = os.path.join(config.VLM_DIR, item["image_path"])
        
        if use_vcot:
            prompt = item["question"] + "\n\nCRITICAL INSTRUCTION: Before you answer, you MUST list the approximate [x, y] coordinates of EVERY single target object you see. Write them out one by one. After listing all coordinates, output a newline, and then provide ONLY the final integer count as your very last line."
        else:
            prompt = item["question"] + "\n\nProvide ONLY the final integer count as your answer. Do not include any other text."
        
        try:
            if is_openai:
                base64_image = encode_image_base64(img_path)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    temperature=0.0,
                )
                pred_text = response.choices[0].message.content.strip().replace(",", "")
            else:
                image_part = file_to_generative_part(img_path, "image/png")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                    )
                )
                pred_text = response.text.strip().replace(",", "")
            
            # Extract just the number
            import re
            
            if use_vcot:
                # For vCoT, we want the LAST number they emit (the final count)
                # Ignore coordinate numbers
                lines = [l.strip() for l in pred_text.split('\n') if l.strip()]
                last_line = lines[-1] if lines else "0"
                numbers = re.findall(r'\d+', last_line)
                pred_count = int(numbers[-1]) if numbers else -1
            else:
                numbers = re.findall(r'\d+', pred_text)
                pred_count = int(numbers[0]) if numbers else -1
            
        except Exception as e:
            print(f"Error on {item['id']}: {e}")
            pred_count = -1
            
        is_correct = (pred_count == item["answer"])
        error = abs(pred_count - item["answer"])
        
        if is_correct:
            correct_count += 1
            
        total_error += error
        diff = item["difficulty"]
        difficulty_stats[diff]["total"] += 1
        if is_correct:
            difficulty_stats[diff]["correct"] += 1
        difficulty_stats[diff]["abs_error_sum"] += error
        
        results.append({
            "id": item["id"],
            "difficulty": diff,
            "ground_truth": item["answer"],
            "prediction": pred_count,
            "error": error,
            "correct": is_correct
        })
        
    # Save detailed results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    # Print Summary
    print("\n" + "="*40)
    print(f"Evaluation Summary: {model_name}")
    print("="*40)
    print(f"Overall Accuracy: {(correct_count / len(annotations))*100:.1f}% ({correct_count}/{len(annotations)})")
    print(f"Overall Mean Absolute Error (MAE): {total_error / len(annotations):.2f}")
    
    print("\nBreakdown by Difficulty:")
    for diff, stats in difficulty_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            mae = stats["abs_error_sum"] / stats["total"]
            print(f"  {diff.capitalize()}:")
            print(f"    Accuracy: {acc:.1f}%")
            print(f"    MAE:      {mae:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM on counting dataset")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", 
                        help="The VLM model to use (e.g. gpt-4o, gemini-2.5-flash)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of images to evaluate per difficulty tier")
    parser.add_argument("--vcot", action="store_true", help="Enable Visual Chain-of-Thought prompting")
    args = parser.parse_args()
    
    evaluate_model(model_name=args.model, max_per_level=args.limit, use_vcot=args.vcot)
