import torch

def preprocess_function(examples, tokenizer, max_length=512, input_column="input", target_column="target"):
    input_texts = examples[input_column]
    target_texts = examples[target_column]
    
    all_tokenized = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "sentence_id": [],
        "task": []
    }

    for i in range(len(input_texts)):
        # Full chat with user + assistant (no generation prompt)
        messages = [
            {"role": "user", "content": input_texts[i]},
            {"role": "assistant", "content": target_texts[i]}
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # ✅ Not needed here
        )
        tokenized = tokenizer(full_prompt, max_length=max_length, padding="max_length", truncation=True)

        # Just the user message to find where assistant reply begins
        user_only_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_texts[i]}],
            tokenize=False,
            add_generation_prompt=True  # ✅ Needed here to mark the cutoff
        )
        user_only_ids = tokenizer(user_only_prompt, max_length=max_length, truncation=True)["input_ids"]

        # Mask the user part in labels
        labels = tokenized["input_ids"][:]
        labels[:len(user_only_ids)] = [-100] * len(user_only_ids)

        # Collect results
        all_tokenized["input_ids"].append(tokenized["input_ids"])
        all_tokenized["attention_mask"].append(tokenized["attention_mask"])
        all_tokenized["labels"].append(labels)
        all_tokenized["sentence_id"].append(examples["sentence_id"][i])
        all_tokenized["task"].append(examples["task_elements"][i])

    # Convert to tensors
    all_tokenized["input_ids"] = torch.tensor(all_tokenized["input_ids"])
    all_tokenized["attention_mask"] = torch.tensor(all_tokenized["attention_mask"])
    all_tokenized["labels"] = torch.tensor(all_tokenized["labels"])
    all_tokenized["sentence_id"] = torch.tensor(all_tokenized["sentence_id"])

    return all_tokenized

def preprocess_function_base(examples, tokenizer, max_length=512, input_column="input", target_column="target"):
    input_texts = examples[input_column]
    target_texts = examples[target_column]
    
    all_tokenized = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "sentence_id": [],
        "task": []
    }

    for i in range(len(input_texts)):
        full_prompt = input_texts[i] + " =>" + target_texts[i]
        tokenized = tokenizer(full_prompt, max_length=max_length, padding="max_length", truncation=True)

        input_only_prompt = input_texts[i] + " =>"
        input_only_ids = tokenizer(input_only_prompt, max_length=max_length, truncation=True)["input_ids"]

        # Mask the user part in labels
        labels = tokenized["input_ids"][:]
        labels[:len(input_only_ids)] = [-100] * len(input_only_ids)

        # Collect results
        all_tokenized["input_ids"].append(tokenized["input_ids"])
        all_tokenized["attention_mask"].append(tokenized["attention_mask"])
        all_tokenized["labels"].append(labels)
        all_tokenized["sentence_id"].append(examples["sentence_id"][i])
        all_tokenized["task"].append(examples["task_elements"][i])

    # Convert to tensors
    all_tokenized["input_ids"] = torch.tensor(all_tokenized["input_ids"])
    all_tokenized["attention_mask"] = torch.tensor(all_tokenized["attention_mask"])
    all_tokenized["labels"] = torch.tensor(all_tokenized["labels"])
    all_tokenized["sentence_id"] = torch.tensor(all_tokenized["sentence_id"])

    return all_tokenized