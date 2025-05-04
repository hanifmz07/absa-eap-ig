import json
from datasets import Dataset

def load_json_dataset(json_path):
    """Load JSON dataset into a Hugging Face Dataset."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = Dataset.from_list(data)
    return data