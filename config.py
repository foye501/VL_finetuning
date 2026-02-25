"""
Configuration for VLM Synthetic Counting Dataset Generation.

Generates images with known object counts for benchmarking
Vision-Language Models on counting tasks.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VLM_DIR = os.path.join(DATA_DIR, "vlm")
IMAGE_DIR = os.path.join(VLM_DIR, "images")
ANNOTATION_FILE = os.path.join(VLM_DIR, "annotations.json")

# ─── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Difficulty Levels ───────────────────────────────────────────────────────
# Each level controls count range, scene complexity, and number of samples
DIFFICULTY_LEVELS = {
    "easy": {
        "count_range": (1, 5),
        "samples": 100,
        "description": "Single object type, clean background, no overlap",
        "allow_overlap": False,
        "max_distractors": 0,
        "backgrounds": ["white", "light_gray"],
    },
    "medium": {
        "count_range": (6, 20),
        "samples": 100,
        "description": "Mixed objects, colored bg, some distractors",
        "allow_overlap": True,
        "max_distractors": 5,
        "backgrounds": ["light_blue", "light_green", "light_yellow", "beige"],
    },
    "hard": {
        "count_range": (21, 50),
        "samples": 100,
        "description": "Dense scenes, significant overlap, many distractors",
        "allow_overlap": True,
        "max_distractors": 15,
        "backgrounds": ["gradient", "noisy", "textured"],
    },
    "extreme": {
        "count_range": (51, 100),
        "samples": 100,
        "description": "Very dense, heavy occlusion, visually challenging",
        "allow_overlap": True,
        "max_distractors": 25,
        "backgrounds": ["gradient", "noisy", "complex"],
    },
}

# ─── Image Settings ──────────────────────────────────────────────────────────
IMAGE_SIZE = (512, 512)  # Width, Height

# ─── Object Definitions ─────────────────────────────────────────────────────
# Shapes the generator can place
SHAPES = ["circle", "square", "triangle", "star", "diamond", "pentagon", "hexagon"]

# Colors for target and distractor objects
TARGET_COLORS = [
    (220, 50, 50),    # Red
    (50, 120, 220),   # Blue
    (50, 180, 50),    # Green
    (220, 180, 30),   # Yellow
    (180, 50, 220),   # Purple
    (220, 130, 30),   # Orange
]

DISTRACTOR_COLORS = [
    (150, 150, 150),  # Gray
    (180, 140, 100),  # Tan
    (100, 140, 180),  # Steel blue
    (140, 180, 100),  # Olive
    (180, 100, 140),  # Mauve
]

# Object size range (radius in pixels)
OBJECT_SIZE_RANGE = {
    "easy":    (25, 45),
    "medium":  (18, 35),
    "hard":    (12, 25),
    "extreme": (8, 18),
}

# ─── Background Colors ──────────────────────────────────────────────────────
BACKGROUND_COLORS = {
    "white":        (255, 255, 255),
    "light_gray":   (240, 240, 240),
    "light_blue":   (220, 235, 255),
    "light_green":  (220, 255, 230),
    "light_yellow": (255, 255, 220),
    "beige":        (245, 235, 220),
}

# ─── Question Templates ─────────────────────────────────────────────────────
QUESTION_TEMPLATES = [
    "How many {color} {shape}s are in this image?",
    "Count the number of {color} {shape}s.",
    "What is the total count of {color} {shape}s in the image?",
    "How many {shape}s of {color} color can you see?",
]

QUESTION_TEMPLATES_NO_COLOR = [
    "How many {shape}s are in this image?",
    "Count the number of {shape}s.",
    "What is the total count of {shape}s in the image?",
    "How many {shape}s can you see?",
]

# ─── Output Format ───────────────────────────────────────────────────────────
# Each annotation entry:
# {
#   "id": str,
#   "image_path": str,          # relative path to image
#   "question": str,
#   "answer": int,              # ground truth count
#   "difficulty": str,          # easy/medium/hard/extreme
#   "target_shape": str,
#   "target_color": str,
#   "target_color_rgb": [R,G,B],
#   "num_distractors": int,
#   "distractor_shapes": [str],
#   "has_overlap": bool,
#   "image_size": [W, H],
#   "object_positions": [...],  # bounding boxes for all targets
# }
