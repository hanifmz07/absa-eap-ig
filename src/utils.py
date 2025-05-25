import torch, re, ast
import pandas as pd
from typing import Optional, List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoConfig
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights, convert_bloom_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from torch.utils.data import Dataset, DataLoader
from eap.graph import Graph
import random
import pickle
import os

# Automatically select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_cfg_dict(base_model_name: str, hf_config) -> dict:
    """
    Generate a TransformerLens config dictionary based on a HuggingFace config.

    Args:
        base_model_name (str): Name of the base model (e.g., "Qwen/Qwen2.5-0.5B").
        hf_config: HuggingFace model config object.

    Returns:
        dict: TransformerLens-compatible config dictionary.
    """
    if "qwen2" in base_model_name.lower():
        return {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": int(hf_config.rope_theta),
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
            "default_prepend_bos": False,
        }
    elif "bloom" in base_model_name.lower():
        return {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu_fast",
            "eps": hf_config.layer_norm_epsilon,
            "normalization_type": "LN",
            "post_embedding_ln": True,
            "positional_embedding_type": "alibi",
            "default_prepend_bos": False,
        }
    raise ValueError(
            f"Unsupported base model: '{base_model_name}'. "
            f"Currently supported: Qwen2, Bloom."
        )

def load_model(base_model_name: str,
               fine_tuned_model_path: Optional[str] = None,
               device: str = device) -> HookedTransformer:
    
    """
    Load either a base TransformerLens model or a fine-tuned HuggingFace model converted to TransformerLens.

    Args:
        base_model_name (str): Name of the base model (e.g., "qwen/Qwen2-0.5B").
        fine_tuned_model_path (Optional[str]): If given, loads and converts a fine-tuned HF model.
        device (str): Device to load model onto.

    Returns:
        HookedTransformer: The loaded TransformerLens model.
    """

    if fine_tuned_model_path:
        print(f"Loading fine-tuned model from HuggingFace at {fine_tuned_model_path}...")

        model = load_finetuned_model(base_model_name, fine_tuned_model_path, device)

    else:
        print(f"Loading base TransformerLens model: {base_model_name}...")
        model = HookedTransformer.from_pretrained(base_model_name, device=device)

    return model
    

def load_finetuned_model(base_model_name: str, 
                         fine_tuned_model_path: str, 
                         device: str = device) -> HookedTransformer:
    """
    Load a fine-tuned HuggingFace Qwen2 model into TransformerLens.

    Args:
        base_model_name (str): Name of the base HuggingFace model.
        fine_tuned_model_path (str): Local path to the fine-tuned model directory.

    Returns:
        HookedTransformer: The converted TransformerLens model.
    """
    # Load fine-tuned HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        fine_tuned_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Load base model config
    hf_config = AutoConfig.from_pretrained(base_model_name)
    cfg = HookedTransformerConfig.from_dict(get_cfg_dict(base_model_name, hf_config))

    # Convert state dict and load into HookedTransformer
    if "qwen2" in base_model_name.lower():
        state_dict = convert_qwen2_weights(hf_model, cfg)
    elif "bloom" in base_model_name.lower():
        state_dict = convert_bloom_weights(hf_model, cfg)

    model = HookedTransformer.from_pretrained(
        model_name=base_model_name,
        device=device
    )
    model.load_and_process_state_dict(state_dict)

    return model

def load_finetuned_model_lens_from_dir(dir: str, device: str = device) -> HookedTransformer:
    """
    Load a fine-tuned TransformerLens model from a specified directory.

    Args:
        dir (str): Directory containing the model files.
        device (str): Device to load the model onto.
    
    Returns:
        HookedTransformer: The loaded TransformerLens model.
    """
    with open(os.path.join(dir, 'model_config.pkl'), 'rb') as f:
        new_cfg_dict = pickle.load(f)
    new_cfg = HookedTransformerConfig.from_dict(new_cfg_dict)
    new_model = HookedTransformer(new_cfg)
    new_model.load_state_dict(torch.load(os.path.join(dir, 'model.pt'), map_location=device))
    return new_model

def calculate_metrics(predictions: List[List[Dict[str, str]]], targets: List[List[Dict[str, str]]], task='') -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for the given predictions and targets for ABSA.

    Args:
        predictions (List[List[Dict[str, str]]]): List of predicted triplets.
        targets (List[List[Dict[str, str]]]): List of target triplets.
        task (str): The task name for which metrics are calculated.
    
    Returns:
        Dict[str, float]: A dictionary containing precision, recall, and F1 score.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction,target in zip(predictions,targets):
        for target_tuple in target:
            if target_tuple in prediction:
                true_positive += 1
            else:
                false_negative += 1
        false_positive += sum(1 for pred in prediction if pred not in target)
    precision = true_positive/(true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive/(true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * recall * precision)/(recall + precision) if (recall + precision) > 0 else 0
    return {
        f"precision_{task}" : precision,
        f"recall_{task}" : recall,
        f"f1_{task}" : f1
    }

def parse_absa_string(text: str) -> List[Dict[str, str]]:
    """
    Parses a string formatted as "[A] aspect [O] opinion [S] sentiment" into a list of dictionaries.
    Each dictionary contains the tag as the key and the corresponding value.
    For example, "[A] [O] [S] [A] harga [O] terjangkau [S] positive [SSEP] [A] fasilitas [O] nyaman [S] positive" becomes:
    [{'A': 'harga', 'S': 'positive', 'O': 'terjangkau'},
    {'A': 'fasilitas', 'S': 'positive', 'O': 'nyaman'}].

    Args:
        text (str): ABSA string output to be parsed.

    Returns:
        List[Dict[str, str]]: List of dictionaries of parsed ABSA output.

    """
    pattern = r"\[(\w+)\]\s*([^[]+)"
    matches = re.findall(pattern, text)

    result = []
    current_dict = {}

    for tag, content in matches:
        if tag == "SSEP":  # Sentence separator -> Start a new dictionary
            result.append(current_dict)
            current_dict = {}
        else:
            current_dict[tag] = content.strip()

    if current_dict:  # Append the last sentence if it exists
        result.append(current_dict)

    return result

def postprocess_absa_outputs(preds: List[str], labels: List[str], sentence_id: List[int], task: List[str]) -> Dict[str, Dict[str, List[List[Dict[str, str]]]]]:
    """
    Aggregate the predictions for each order of elements in the triplet into one output.

    Args:
        preds (List[str]): List of predicted strings.
        labels (List[str]): List of label strings.
        sentence_id (List[int]): List of sentence IDs.
        task (List[str]): List of task elements of the ABSA.
    
    Returns:
        Dict[str, Dict[str, List[List[Dict[str, str]]]]]: A dictionary where the keys are task names and the values are dictionaries
            containing predictions and targets for each sentence ID.
    """
    per_task1 = {}
    for p, l, si, t in zip(preds, labels, sentence_id, task):
        if t not in per_task1:
            per_task1[t] = {}
        if si not in per_task1[t]:
            per_task1[t][si] = {
                "m" : 0, # from mvp
                "preds" : [],
                "labels" : parse_absa_string(l)
            }
        per_task1[t][si]["preds"].extend(parse_absa_string(p))
        per_task1[t][si]["m"] += 1
    per_task2 = {}
    for t, v1 in per_task1.items():
        if t not in per_task2:
            per_task2[t] = {
                "predictions" : [],
                "targets" : []
            }
        for si, v2 in v1.items():
            m = v2["m"]
            unique_preds = []
            for el in v2["preds"]:
                if el not in unique_preds:
                    unique_preds.append(el)
            aggregated = []
            for el in unique_preds:
                if v2["preds"].count(el) >= m/2:
                    aggregated.append(el)
            per_task2[t]["predictions"].append(aggregated)
            per_task2[t]["targets"].append(v2["labels"])
    return per_task2



class ABSAAutoRegressiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, shuffle=False, seed=42):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Concatenate prompt and label into one sequence
        full_text = sample["input"].strip() + " " + sample["target"].strip()

        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"  
        )
        
        tokens = encoding["input_ids"].squeeze()
        return {
            "tokens": tokens
        }

def apply_active_edge_unfreezing(model, csv_path: str) -> None:
    """
    Apply selective unfreezing and gradient masking to a TransformerLens model
    based on active attention and MLP nodes from an edge CSV file.

    This function:
    - Parses active attention heads and MLP layers from the CSV.
    - Applies gradient masking to inactive heads in attention projection matrices (W_Q, W_K, W_V, W_O).
    - Unfreezes MLP layers, embedding, and unembedding weights.

    Args:
        model: TransformerLens model instance.
        csv_path (str): Path to CSV file with 'child_node' and 'child_type' columns.
    """
    df = pd.read_csv(csv_path)
    attention_targets = set()
    mlp_layers = set()
    heads_per_layer = {}

    for _, row in df.iterrows():
        child_node = row.get("child_node")
        child_type = row.get("child_type")

        if isinstance(child_node, str):
            if child_node.startswith("a") and ".h" in child_node:
                try:
                    layer_str, head_str = child_node[1:].split(".h")
                    layer = int(layer_str)
                    head = int(head_str)
                    attention_targets.add((layer, head, child_type))
                    heads_per_layer.setdefault(layer, set()).add(head)
                except ValueError:
                    print(f"Warning: Could not parse attention node: {child_node}")
            elif child_node.startswith("m"):
                try:
                    layer = int(child_node[1:])
                    mlp_layers.add(layer)
                except ValueError:
                    print(f"Warning: Could not parse MLP node: {child_node}")

    # Freeze all registered buffers (e.g., positional embeddings)
    for _, buffer in model.named_buffers():
        if isinstance(buffer, torch.Tensor):
            buffer.requires_grad = False

    def register_head_mask(weight_tensor: torch.Tensor, active_heads: list[int]) -> None:
        """
        Register a backward hook that masks gradients for inactive heads in the given weight tensor.

        Args:
            weight_tensor (torch.Tensor): The attention weight matrix (e.g., W_Q, W_K).
            active_heads (list[int]): List of head indices to remain trainable.
        """
        mask = torch.zeros_like(weight_tensor)
        for head in active_heads:
            mask[head] = 1.0

        def mask_hook(grad):
            return grad * mask

        weight_tensor.register_hook(mask_hook)

    # Apply gradient masks to W_Q, W_K, W_V
    for proj_type in ['q', 'k', 'v']:
        layer_to_heads = {
            layer: [head for (l, head, t) in attention_targets if l == layer and t == proj_type]
            for layer in set(l for (l, _, t) in attention_targets if t == proj_type)
        }

        for layer, heads in layer_to_heads.items():
            if proj_type == 'q':
                register_head_mask(model.blocks[layer].attn.W_Q, heads)
            elif proj_type == 'k':
                register_head_mask(model.blocks[layer].attn.W_K, heads)
            elif proj_type == 'v':
                register_head_mask(model.blocks[layer].attn.W_V, heads)

    # Apply gradient masks to W_O for all heads involved in any Q/K/V projection
    for layer, heads in heads_per_layer.items():
        register_head_mask(model.blocks[layer].attn.W_O, heads)

    # Unfreeze selected MLP layers
    for layer in mlp_layers:
        model.blocks[layer].mlp.W_in.requires_grad = True
        model.blocks[layer].mlp.W_out.requires_grad = True
        if hasattr(model.blocks[layer].mlp, "W_gate"):
            model.blocks[layer].mlp.W_gate.requires_grad = True

    # Unfreeze embedding and unembedding weights
    model.embed.W_E.requires_grad = True
    model.unembed.W_U.requires_grad = True
    model.unembed.b_U.requires_grad = True


def convert_triplet_string(triplet_str: str) -> tuple:
    """
    Safely parses a stringified triplet like '[("aspect", "opinion", "sentiment")]'
    and returns the individual components.

    Returns:
        Tuple of (aspect, opinion, sentiment) or empty strings if invalid.
    """
    try:
        triplet = ast.literal_eval(triplet_str)[0]
        return tuple(triplet)
    except (ValueError, SyntaxError, IndexError):
        return "", "", ""


def format_by_mode(aspect: str, opinion: str, sentiment: str, mode: str = "AOS") -> str:
    """
    Formats the triplet based on the selected mode.

    Args:
        aspect, opinion, sentiment: Components of the triplet.
        mode (str): One of "A", "O", "S", or "AOS".

    Returns:
        A formatted string containing only the selected tags and values.
    """
    parts = []
    if "A" in mode:
        parts.append(f"[A] {aspect}")
    if "O" in mode:
        parts.append(f"[O] {opinion}")
    if "S" in mode:
        parts.append(f"[S] {sentiment}")
    return " ".join(parts)


def build_suffix_from_mode(mode: str) -> str:
    """
    Builds the prompt suffix based on mode, e.g. "[A] [S]"

    Args:
        mode (str): One of "A", "O", "S", or "AOS"

    Returns:
        A space-separated suffix string.
    """
    mapping = {"A": "[A]", "O": "[O]", "S": "[S]"}
    # return " " + " ".join([mapping[c] for c in mode if c in mapping])
    return ' [A] [O] [S]'


def extract_by_mode(text: str, mode: str) -> str:
    """
    Extracts the relevant generated fields based on the selected mode.

    Args:
        text (str): The raw model output.
        mode (str): One of "A", "O", "S", or "AOS".

    Returns:
        A cleaned string containing only the generated fields of interest.
    """
    result = []
    for tag in mode:
        match = re.search(rf"\[{tag}\]\s*(.*?)\s*(?=\[|$)", text)
        result.append(f"[{tag}] {match.group(1).strip()}" if match else f"[{tag}]")
    return " ".join(result).strip()


def filter_correct_data(
    model,
    data: pd.DataFrame,
    sentence_col: str,
    label_col: str,
    max_tokens: int = 50,
    filter_only_correct: bool = True,
    filter_mode: str = "AOS",
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters a dataset based on whether the model correctly generates the target aspect,
    opinion, sentiment, or full triplet depending on the specified mode.

    Args:
        model: A TransformerLens HookedTransformer model with `.generate()` method.
        data (pd.DataFrame): DataFrame with sentences and triplet labels.
        sentence_col (str): Column name containing the sentence inputs.
        label_col (str): Column name containing stringified triplet labels.
        max_tokens (int): Maximum number of tokens to generate.
        filter_only_correct (bool): If True, only return matching predictions.
        filter_mode (str): Must be one of "A", "O", "S", or "AOS".
        save_path (Optional[str]): If given, saves filtered results to CSV.

    Returns:
        pd.DataFrame: DataFrame with new columns: 'original_label', 'inference', and 'is_match'.
    """
    valid_modes = {"A", "O", "S", "AOS"}
    filter_mode = filter_mode.upper()
    if filter_mode not in valid_modes:
        raise ValueError(f"Invalid filter_mode '{filter_mode}'. Must be one of {valid_modes}.")

    suffix = build_suffix_from_mode(filter_mode)
    inputs = data[sentence_col].tolist()
    labels = data[label_col].tolist()

    results, expected_labels, match_flags = [], [], []

    for prompt, label_raw in zip(inputs, labels):
        full_prompt = prompt + suffix
        output = model.generate(
            input=full_prompt,
            max_new_tokens=max_tokens,
            stop_at_eos=True,
            do_sample=False,
            return_type="str"
        )

        # Remove special tokens and clean up
        raw_output = re.sub(r"<\|endoftext\|>", "", output)
        if "[SSEP]" in raw_output:
            raw_output = raw_output.split("[SSEP]", 1)[-1].strip()
        elif raw_output.count("[A]") > 1:
            raw_output = raw_output.split("[A] [O] [S]", 1)[-1].strip()
        else:
            raw_output = raw_output.split(".", 1)[-1].strip()

        # Compare model output vs ground truth
        aspect, opinion, sentiment = convert_triplet_string(label_raw)
        expected = format_by_mode(aspect, opinion, sentiment, mode=filter_mode)
        cleaned = extract_by_mode(raw_output, filter_mode)
        is_match = cleaned.strip() == expected.strip()

        #debugs
        # print(full_prompt)
        # print(output)
        # print(cleaned)
        # print(expected)

        expected_labels.append(expected)
        results.append(cleaned)
        match_flags.append(is_match)

    df_result = data.copy()
    df_result["original_label"] = expected_labels
    df_result["inference"] = results
    df_result["is_match"] = match_flags

    total = len(df_result)
    correct = df_result["is_match"].sum()
    print(f"Correct: {correct} / {total} ({correct / total:.2%}) with mode [{filter_mode}]")

    if filter_only_correct:
        df_result = df_result[df_result["is_match"]].reset_index(drop=True)

    if save_path:
        df_result.to_csv(save_path, index=False)

    return df_result



def append_labels(model: HookedTransformer, clean: List[str], corrupted: List[str], labels: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[str], List[str]]:
    """
    For multi-token labels, add the labels (until second to last token) to clean and corrupted sentences

    Args:
        model (HookedTransformer): The model used for EAP.
        clean: list of clean sentence strings
        corrupted: list of corrupted sentences
        labels: tu

    Returns:
        clean and corrupted sentences with gold target sentence appended.
    """
    new_clean, new_corrupted = [], []
    for i in range(len(clean)):
        new_clean.append(model.to_string([*model.to_tokens(clean[i]).squeeze(), *labels[0][i][:-1]]))
        new_corrupted.append(model.to_string([*model.to_tokens(corrupted[i]).squeeze(), *labels[1][i][:-1]]))
    return new_clean, new_corrupted


def build_eap_dataset(
    model: HookedTransformer,
    df: pd.DataFrame,
    sentence_col: str,
    triplet_col: str,
    corrupted_col: str,
    corrupted_triplet_col: str,
    suffix: str,
    filer_same_length_counterfactuals: bool = True,
    append_labels= False,
) -> pd.DataFrame:
    """
    Builds an EAP dataset from a filtered dataframe for use with EAP-IG,
    saving both token ids and string values of correct/incorrect labels.

    Args:
        model (HookedTransformer): TransformerLens model used for tokenization.
        df (pd.DataFrame): DataFrame containing sentence and counterfactual data.
        sentence_col (str): Column name for the original sentence.
        triplet_col (str): Column name for the original triplet string.
        corrupted_col (str): Column name for the corrupted sentence.
        corrupted_triplet_col (str): Column name for the corrupted triplet string.
        suffix (str): Prompt suffix to add (e.g., "[A]").
        filer_same_length_counterfactuals (bool): If True, remove datapoints where
            counterfactual token length differs from original token length.

    Returns:
        pd.DataFrame: DataFrame with clean/corrupted prompts, token indices, and raw label texts.
    """

    if suffix == "[A]":
        idx = 0
    elif suffix == "[O]":
        idx = 1
    elif suffix == "[S]":
        idx = 2
    else:
        raise ValueError(f"Invalid suffix '{suffix}'. Must be one of '[A]', '[O]', or '[S]'.")

    eap_data = []
    num_removed = 0
    for _, row in df.iterrows():
        if row["is_match"]:
            clean = row[sentence_col] + f" {suffix}"
            corrupted = row[corrupted_col] + f" {suffix}"

            if filer_same_length_counterfactuals:
                clean_tokens = model.to_tokens(clean)
                corrupted_tokens = model.to_tokens(corrupted)
                if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
                    num_removed += 1
                    continue

            try:
                original_triplet = ast.literal_eval(row[triplet_col])
                corrupted_triplet = ast.literal_eval(row[corrupted_triplet_col])

                correct_label = original_triplet[0][idx]
                incorrect_label = corrupted_triplet[0][idx]

                correct_idx = model.to_tokens(f" {correct_label}").tolist()
                incorrect_idx = model.to_tokens(f" {incorrect_label}").tolist()

                if len(correct_idx[0]) != len(incorrect_idx[0]):
                    num_removed += 1
                    continue

                # if label has multiple tokens, append all but last label tokens to clean and corrupted
                # because we need to obtain the right logit conditioned on the correct prefix
                if append_labels:
                    if len(correct_idx[0]) > 1: # TODO
                        clean += model.to_string(correct_idx[0][:-1])
                        corrupted += model.to_string(incorrect_idx[0][:-1])

                eap_data.append({
                    "clean": clean,
                    "corrupted": corrupted,
                    "correct_label": correct_label,
                    "incorrect_label": incorrect_label,
                    "correct_idx": correct_idx,
                    "incorrect_idx": incorrect_idx,
                })
            except Exception as e:
                print(f"Skipping row due to parsing/tokenizing error: {e}")
                continue

    if filer_same_length_counterfactuals:
        print(f"Removed {num_removed} out of {len(df)} datapoints that does not match token length.")
    print(f"Filtered data size {len(eap_data)=}")
    return pd.DataFrame(eap_data)

  
def safe_parse(raw):
    try:
        return ast.literal_eval(raw)[0]  # unbox the list-of-list
        # return ast.literal_eval(raw)[0][0]  # unbox the list-of-list
    except Exception as e:
        raise ValueError(f"Failed to parse: {raw}\n{e}")
    
def collate_EAP(batch):
    clean, corrupted, labels = zip(*batch)

    correct_idx_batch = [torch.tensor(l[0], dtype=torch.long) for l in labels]
    incorrect_idx_batch = [torch.tensor(l[1], dtype=torch.long) for l in labels]

    return list(clean), list(corrupted), (correct_idx_batch, incorrect_idx_batch)

class EAPDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clean = row["clean"]
        corrupted = row["corrupted"]
        correct_idx = safe_parse(row["correct_idx"])
        incorrect_idx = safe_parse(row["incorrect_idx"])
        return clean, corrupted, [correct_idx, incorrect_idx]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP, drop_last=False)


def edge_merging(graph_paths: List[str]) -> pd.DataFrame:
    edge_set = set()

    data_dict = {
        "parent_node": [],
        "child_node": [],
        "child_type": [],
    }

    for gp in graph_paths:
        graph = Graph.from_pt(gp)
        for edge in graph.edges.values():
            if edge.in_graph:
                edge_tuple = (edge.parent.name, edge.child.name, edge.qkv)
                if edge_tuple not in edge_set:
                    data_dict["parent_node"].append(edge.parent.name)
                    data_dict["child_node"].append(edge.child.name)
                    data_dict["child_type"].append(edge.qkv)
                    edge_set.add(edge_tuple)

    return pd.DataFrame(data_dict)
