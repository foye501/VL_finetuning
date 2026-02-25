import os
import sys
import json
from collections import Counter

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze_dataset():
    """Analyze the generated VLM counting dataset."""
    if not os.path.exists(config.ANNOTATION_FILE):
        print(f"Error: Annotation file not found at {config.ANNOTATION_FILE}")
        return
        
    with open(config.ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)
        
    print(f"Total images generated: {len(annotations)}")
    
    # Basic Stats
    difficulty_counts = Counter([a["difficulty"] for a in annotations])
    print("\n--- Samples by Difficulty ---")
    for diff, count in difficulty_counts.items():
        print(f"  {diff.capitalize()}: {count}")
        
    # Analyze ground truth counts
    print("\n--- Answer Distributions ---")
    for diff in config.DIFFICULTY_LEVELS.keys():
        diff_anns = [a for a in annotations if a["difficulty"] == diff]
        if not diff_anns:
            continue
            
        answers = [a["answer"] for a in diff_anns]
        avg_answer = sum(answers) / len(answers)
        min_answer = min(answers)
        max_answer = max(answers)
        
        target_range = config.DIFFICULTY_LEVELS[diff]["count_range"]
        
        print(f"{diff.capitalize()}:")
        print(f"  Target Range: {target_range}")
        print(f"  Actual Range: {min_answer} - {max_answer}")
        print(f"  Average Count: {avg_answer:.1f}")
        
        if max_answer < target_range[1] * 0.8 and diff in ["hard", "extreme"]:
            print(f"  ⚠️ WARNING: Max actual count ({max_answer}) is significantly lower than target max ({target_range[1]}). Scene density might be preventing object placement.")
            
    # Overlap Stats
    print("\n--- Overlap Statistics ---")
    overlap_counts = Counter([a["has_overlap"] for a in annotations])
    print(f"  Images with overlap: {overlap_counts[True]} ({(overlap_counts[True]/len(annotations))*100:.1f}%)")
    print(f"  Images without overlap: {overlap_counts[False]} ({(overlap_counts[False]/len(annotations))*100:.1f}%)")
    
    # Distractor Stats
    print("\n--- Distractor Statistics ---")
    distractors = [a["num_distractors"] for a in annotations]
    print(f"  Average distractors per image: {sum(distractors)/len(distractors):.1f}")
    print(f"  Max distractors in a single image: {max(distractors)}")

    # Verify positions
    print("\n--- Ground Truth Verification ---")
    errors = 0
    for a in annotations:
        if len(a["object_positions"]) != a["answer"]:
            print(f"  ⚠️ Error in {a['id']}: Answer is {a['answer']} but {len(a['object_positions'])} bounding boxes saved!")
            errors += 1
            
    if errors == 0:
        print("  ✅ All bounding box counts match the ground truth answers.")
        
    print("\nDataset validation complete.")

if __name__ == "__main__":
    analyze_dataset()
