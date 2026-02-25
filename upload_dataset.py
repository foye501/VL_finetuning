import os
import json
from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi

# ==============================================================================
# Upload Dataset to Hugging Face
# ==============================================================================

# Ensure you are logged in to Hugging Face:
# run `huggingface-cli login` in your terminal before running this script.

DATA_DIR = "./data"
JSON_FILE = os.path.join(DATA_DIR, "vlm_counting_train.json")
HF_REPO_ID = "foye501/VLM-Counting-dataset" # Change this to your HF username/repo

def main():
    print(f"Loading LLaMA-Factory dataset from {JSON_FILE}...")
    
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
        
    # We need to restructure the data slightly for the HuggingFace datasets library
    # LLaMA-Factory format: {"messages": [...], "images": ["path/to/img.png"]}
    
    formatted_data = {
        "messages": [item["messages"] for item in data],
        "image": [item["images"][0] for item in data] # datasets library handles Image loading better as singular 'image' column usually
    }
    
    print("Formatting into HuggingFace Dataset object...")
    
    # Define features to ensure the image path is cast to an actual PIL Image on upload
    features = Features({
        "messages": [{"content": Value("string"), "role": Value("string")}],
        "image": Image()
    })
    
    hf_dataset = Dataset.from_dict(formatted_data, features=features)
    
    print(f"Dataset mapped successfully: {len(hf_dataset)} rows.")
    print(f"Pushing to HuggingFace Hub: {HF_REPO_ID} (This will take a while for 10,000 images)...")
    
    try:
        # Pushing to the Hub. `private=True` keeps it secure.
        hf_dataset.push_to_hub(HF_REPO_ID, private=True)
        print(f"\nSuccess! Dataset uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")
        print("You can now download this on your remote server!")
    except Exception as e:
        print("\nERROR pushing to Hugging Face:")
        print(e)
        print("\nDid you run `huggingface-cli login` first?")

if __name__ == "__main__":
    main()
