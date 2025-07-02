import json
from datasets import Dataset
from datasets import concatenate_datasets
import re
import copy

def parse_target(text):
    pattern = r"\[A\] (.*?) \[O\] (.*?) \[S\] (.*?)(?: \[SSEP\]|$)"
    matches = re.findall(pattern, text)
    return matches

def strip_aos(text):
    text = text.replace("[A] [O] [S]", "")
    text = text.strip()
    return text

def open_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure it's a list of dictionaries
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("JSON file must contain a list of dictionaries.")
    
    # Filter for items with element_order == 'aos'
    filtered_data = [item for item in data if item.get("element_order") == "aos"]
    
    data = Dataset.from_list(filtered_data)
    data = data.map(
        lambda row: {
            "input" : strip_aos(row["input"]),
            "target" : parse_target(row["target"])
        },
        batched=False
    )

    return data

def preprocess_(row, order):
    order_map = {
        'a' : 0,
        'o' : 1,
        's' : 2
    }

    prompt = []
    for el in order:
        prompt.append(f"[{el.upper()}]")
    prompt = " ".join(prompt)

    input_text = row["input"] + " " + prompt

    output_text = []

    for tup in row["target"]:
        tup_text = []
        for el in order:
            tup_text.append(f"[{el.upper()}]")
            tup_text.append(tup[order_map[el]])
        tup_text = " ".join(tup_text)
        output_text.append(tup_text)
    
    output_text = " [SSEP] ".join(output_text)

    return input_text + " " + output_text

def preprocess_data(data, tokenizer, order_list=["ao", "os"]):
    all_dataset = []
    for order in order_list:
        new_data = copy.deepcopy(data)
        column_names = new_data.column_names
        new_data = new_data.map(
            lambda row: {
                "text" : preprocess_(row, order)
            },
            batched=False,
            remove_columns=column_names
        )
        all_dataset.append(new_data)
    
    all_dataset = concatenate_datasets(all_dataset)

    def tokenize(examples):
        return tokenizer(
            [" ".join(x) for x in examples["text"]],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    all_dataset = all_dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    return all_dataset