import torch, re, ast
import pandas as pd
from typing import Optional, List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoConfig
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights, convert_bloom_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from torch.utils.data import Dataset, DataLoader
from eap.graph import Graph
from ast import literal_eval
from copy import deepcopy
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
    def __init__(self, data, tokenizer, max_len=300, shuffle=False, seed=42, sample_size=None):
        if shuffle:
            random.seed(seed)
            random.shuffle(data)
        if sample_size is not None:
            data = data[:sample_size]
        
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        full_text = sample["input"].strip() + " " + sample["target"].strip() + self.tokenizer.eos_token
        encoding = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        tokens = encoding["input_ids"].squeeze()

        # Replace last token with eos if truncated
        if tokens[-1] != self.tokenizer.eos_token_id and tokens[-1] != self.tokenizer.pad_token_id:
            tokens = torch.cat([tokens[:-1], torch.tensor([self.tokenizer.eos_token_id])], dim=0)

        return {
            "tokens": tokens
        }

def get_random_nodes(model, df: pd.DataFrame, *,
                     include_mlp: bool = False,
                     include_logits: bool = False,
                     sample_n: Optional[int] = None,
                     random_state: Optional[int] = None,
                     strict: bool = False) -> pd.DataFrame:
    n_layers = model.cfg.n_layers
    n_heads  = model.cfg.n_heads

    # base head names (no q/k/v suffix yet)
    head_names = [f"a{l}.h{h}" for l in range(n_layers) for h in range(n_heads)]

    # expand into q/k/v
    qkv_heads = head_names * 3
    qkv_types = (["q"] * len(head_names)) + (["k"] * len(head_names)) + (["v"] * len(head_names))

    rows = {"child_node": qkv_heads, "child_type": qkv_types}

    if include_mlp:
        rows["child_node"] += [f"m{l}" for l in range(n_layers)]
        rows["child_type"] += [None] * n_layers

    if include_logits:
        rows["child_node"] += ["logits"]
        rows["child_type"] += [None]

    df_all = pd.DataFrame(rows)

    # strict: exclude entire heads present in df regardless of q/k/v
    if strict and not df.empty:
        blocked_heads = set()
        for cn in df.get("child_node", []):
            if isinstance(cn, str) and cn.startswith("a") and ".h" in cn:
                blocked_heads.add(cn)
        if blocked_heads:
            df_all = df_all[~df_all["child_node"].isin(blocked_heads)]

    key = ["child_node", "child_type"]

    def norm_keys(d: pd.DataFrame) -> pd.DataFrame:
        out = d[key].copy()
        out = out.astype("string").fillna("__NA__")
        return out

    # anti-join
    mask = ~pd.MultiIndex.from_frame(norm_keys(df_all)).isin(
        pd.MultiIndex.from_frame(norm_keys(df))
    )
    result = df_all[mask].copy().reset_index(drop=True)

    if sample_n is not None and len(result) > 0:
        actual_n = min(sample_n, len(result))
        if actual_n < sample_n:
            print(f"[RandomCircuit][Strict] Requested {sample_n} but only {actual_n} available after strict filtering; using {actual_n}.")
        result = result.sample(n=actual_n, random_state=random_state).reset_index(drop=True)

    return result


def apply_active_edge_unfreezing(model, 
                                csv_path: str, 
                                random_circuit: bool = False,
                                sample_n: Optional[int] = None,
                                random_state: Optional[int] = None,
                                strict: bool = False,
                                sample_like_topk: Optional[int] = None):
    import re, os

    def _derive_like_path(path: str, like_topk: int) -> str:
        return re.sub(r"(topk-)\d+", rf"\g<1>{like_topk}", path)

    def _count_unique_pairs_from_csv(path: str) -> int:
        ref = pd.read_csv(path)
        if {"child_node","child_type"}.issubset(ref.columns):
            return ref.drop_duplicates(subset=["child_node","child_type"]).shape[0]
        return len(ref)

    like_used = None
    if random_circuit and sample_like_topk is not None and sample_n is None:
        like_path = _derive_like_path(csv_path, sample_like_topk)
        if os.path.exists(like_path):
            like_n = _count_unique_pairs_from_csv(like_path)
            if like_n > 0:
                sample_n = like_n
                like_used = like_path
                print(f"[RandomCircuit] sample_like_topk={sample_like_topk} → '{like_path}' → unique pairs = {sample_n}")
            else:
                print(f"[RandomCircuit] '{like_path}' has no rows; falling back to provided sample_n={sample_n}")
        else:
            print(f"[RandomCircuit] sibling CSV not found for topk-{sample_like_topk}: {like_path}")

    df = pd.read_csv(csv_path)
    attention_targets = set()
    mlp_layers = set()
    if random_circuit:
        print(f"[Circuit Mode] Using RANDOM circuit (sample_n={sample_n}, random_state={random_state}, strict={strict})")
        if like_used:
            print(f"[Circuit Size] Mirroring size from: {like_used}")
        df_circuit = get_random_nodes(model, df, sample_n=sample_n, random_state=random_state, strict=strict)
        print(f"[Random] Rows used={len(df_circuit)}")
    else:
        print("[Circuit Mode] Using ACTUAL circuit from CSV")
        df_circuit = df.copy()

    for _, row in df_circuit.iterrows():
        child_node = row.get("child_node")
        child_type = row.get("child_type")

        if isinstance(child_node, str):
            if child_node.startswith("a") and ".h" in child_node:
                try:
                    layer_str, head_str = child_node[1:].split(".h")
                    layer = int(layer_str)
                    head = int(head_str)
                    attention_targets.add((layer, head, child_type))
                except ValueError:
                    print(f"Warning: Could not parse attention node: {child_node}")
            elif child_node.startswith("m"):
                try:
                    layer = int(child_node[1:])
                    mlp_layers.add(layer)
                except ValueError:
                    print(f"Warning: Could not parse MLP node: {child_node}")

    # Quick summary
    q_count = sum(1 for (_, _, t) in attention_targets if t == 'q')
    k_count = sum(1 for (_, _, t) in attention_targets if t == 'k')
    v_count = sum(1 for (_, _, t) in attention_targets if t == 'v')
    uniq_heads = len({(l, h) for (l, h, _) in attention_targets})
    print(f"[Parsed] unique_attn_heads={(uniq_heads)} | q={q_count}, k={k_count}, v={v_count} | mlp_layers={sorted(mlp_layers) if mlp_layers else []}")

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


    num_layers = model.cfg.n_layers

    # Apply gradient masks to W_Q, W_K, W_V
    for proj_type in ['q', 'k', 'v']:
        for layer in range(num_layers):
            heads = [head for (l, head, t) in attention_targets if l == layer and t == proj_type]
            if proj_type == 'q':
                register_head_mask(model.blocks[layer].attn.W_Q, heads)
            elif proj_type == 'k':
                register_head_mask(model.blocks[layer].attn.W_K, heads)
            elif proj_type == 'v':
                register_head_mask(model.blocks[layer].attn.W_V, heads)


    heads_per_layer = {layer: set() for layer in range(num_layers)}

    for (layer, head, _) in attention_targets:
        heads_per_layer[layer].add(head)

    # Apply gradient masks to W_O for all heads involved in any Q/K/V projection
    for layer, heads in heads_per_layer.items():
        register_head_mask(model.blocks[layer].attn.W_O, heads)

    print(f"[Done] Gradient masks registered.\n")
    return model


def convert_triplet_string(triplet_str: str) -> tuple:
    """
    Safely parses a stringified triplet like '[("aspect", "opinion", "sentiment")]'
    and returns the individual components.

    Returns:
        Tuple of (aspect, opinion, sentiment) or empty strings if invalid.
    """
    try:
        triplet = ast.literal_eval(triplet_str)
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

def extract_triplet_fixed(text):
    try:
        matches = list(re.finditer(r"\[([AOS])\]", text))
        if len(matches) >= 3:
            a_start = matches[0].end()
            o_start = matches[1].end()
            s_start = matches[2].end()
            aspect = text[a_start:matches[1].start()].strip()
            opinion = text[o_start:matches[2].start()].strip()
            sentiment = text[s_start:].split()[0].strip()
            return (aspect, opinion, sentiment)
    except:
        return None

def filter_correct_data(
    model,
    data: pd.DataFrame,
    sentence_col: str,
    label_col: str,
    max_tokens: int = 150,
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

    suffix = ' [A] [O] [S]'
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

        # Handle multiple triplets
        is_match = True
        triplets_str = raw_output.split("[A] [O] [S]")[-1].strip()
        triplets_str_temp = triplets_str.split("[SSEP]")
        # print(f"Triplets string: {triplets_str}")
        triplets_str_temp = [i.strip() for i in triplets_str_temp]
        labels = convert_triplet_string(label_raw)
        # print(f'Labels: {labels}')
        for triplet_str in triplets_str_temp:
            triplet = extract_triplet_fixed(triplet_str)
            if triplet not in labels:
                print(f"Mismatch found: {triplet} not in {labels}")
                is_match = False
                break
        
        formatted_labels = [f"[A] {label[0]} [O] {label[1]} [S] {label[2]}" for label in labels]
        formatted_labels = " [SSEP] ".join(formatted_labels)
        if is_match:
            results.append(formatted_labels) # Same ordering as the formatted labels
        else:
            results.append(triplets_str)
        expected_labels.append(formatted_labels)
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

def get_tag_suffix(order):
    """
    Generate a tag suffix string from a sequence order.
    
    Args:
        order: String representing the order of tags (e.g., "AOS")
        
    Returns:
        str: Space-separated tag suffix (e.g., "[A] [O] [S]")
    """
    return " ".join(f"[{ch}]" for ch in order)

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
        suffix (str): Prompt suffix to add (e.g., "[A]"). Use "[A] [O] [S]" for the full simultaneous AOS dataset.
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
    elif suffix == "[A] [O] [S]":
        assert "order" in df.columns, "Column 'order' must exist in the DataFrame"
        idx = False
    else:
        raise ValueError(f"Invalid suffix '{suffix}'. Must be one of '[A]', '[O]', '[S]' or '[A] [O] [S]'.")

    eap_data = []
    num_removed = 0
    for _, row in df.iterrows():
        if row["is_match"] and type(row[corrupted_col]) == str:

            if not idx:
                suffix = get_tag_suffix(row["order"])

            clean = row[sentence_col] + f" {suffix}"
            corrupted = row[corrupted_col] + f" {suffix}"

            if filer_same_length_counterfactuals:
                clean_tokens = model.to_tokens(clean)
                corrupted_tokens = model.to_tokens(corrupted)
                if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
                    num_removed += 1
                    continue

            try:
                if not idx:
                    correct_label = row[triplet_col]
                    incorrect_label = row[corrupted_triplet_col]
                
                else: # TODO: Cannot yet handle multiple triplets in the same row
                    original_triplet = ast.literal_eval(row[triplet_col])
                    corrupted_triplet = ast.literal_eval(row[corrupted_triplet_col])
                    
                    correct_label = original_triplet[0][idx]
                    incorrect_label = corrupted_triplet[0][idx]

                correct_idx = model.to_tokens(f" {correct_label}").tolist()
                incorrect_idx = model.to_tokens(f" {incorrect_label}").tolist()

                if len(correct_idx[0]) != len(incorrect_idx[0]):
                    num_removed += 1
                    continue

                # # if label has multiple tokens, append all but last label tokens to clean and corrupted
                # # because we need to obtain the right logit conditioned on the correct prefix
                # if append_labels:
                #     if len(correct_idx[0]) > 1: # TODO
                #         clean += model.to_string(correct_idx[0][:-1])
                #         corrupted += model.to_string(incorrect_idx[0][:-1])

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
    
    eap_df = pd.DataFrame(eap_data)
    eap_df["correct_idx"] = eap_df["correct_idx"].apply(str)
    eap_df["incorrect_idx"] = eap_df["incorrect_idx"].apply(str)    
    return eap_df

def safe_parse(raw):
    """
    Safely parse raw data that could be a list, string, or other format.
    
    Args:
        raw: Raw data to parse, could be list, string, etc.
        
    Returns:
        Parsed data as the first element if it's a list/tensor
        
    Raises:
        ValueError: If parsing fails
    """
    try:
        # If already a list of lists (or torch tensor), just return it
        if isinstance(raw, list):
            return raw[0]
        if isinstance(raw, str):
            return ast.literal_eval(raw)[0]
        raise ValueError(f"Unsupported type for parsing: {type(raw)}")
    except Exception as e:
        raise ValueError(f"Failed to parse: {raw}\n{e}")

    
def collate_EAP(batch):
    """
    Collate function for EAP dataset batching.
    
    Args:
        batch: Batch of data samples from EAPDataset
        
    Returns:
        Tuple containing lists of clean sentences, corrupted sentences, and label tensors
    """
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

# TODO: Change to handle multiple triplets in the same row
def create_full_AOS_dataset(dataset_path):
    """
    Create a full AOS dataset by combining aspects from one counterfactual with opinions/sentiments from another.
    
    Args:
        dataset_path (str): Path to the input CSV dataset
        
    Returns:
        pd.DataFrame: DataFrame with new counterfactual combinations
    """
    df = pd.read_csv(dataset_path)
    
    new_modified_texts = []
    new_modified_triplets = []
    
    for i, row in df.iterrows():
        try:
    
            triplet1 = ast.literal_eval(row['counterfact_triplet1_modified'])
            aspect1 = triplet1[0][0]
    
            triplet3 = ast.literal_eval(row['counterfact_triplet3_modified'])
            _, opinion3, sentiment3 = triplet3[0]
    
            # Replace aspect in sentence (assume it's the first word that matches the original aspect)
            old_aspect3 = triplet3[0][0]
            modified_text = row['counterfact3_modified'].replace(old_aspect3, aspect1, 1)
            modified_triplet = [(aspect1, opinion3, sentiment3)]
    
            new_modified_texts.append(modified_text)
            new_modified_triplets.append(str(modified_triplet))
        except Exception as e:
            print(f"Error in row {i}: {e}")
            new_modified_texts.append("")
            new_modified_triplets.append("")
    
    df["counterfact4_replaced"] = new_modified_texts
    df["counterfact_triplet4_replaced"] = new_modified_triplets

    return df

# Function to construct sequence variants
def create_sequences(a, o, s):
    """
    Create sequence variants for aspect, opinion, and sentiment in different orders.
    
    Args:
        a (str): Aspect text
        o (str): Opinion text  
        s (str): Sentiment text
        
    Returns:
        dict: Dictionary mapping order names to formatted sequences
    """
    return {
        "AOS": f"[A] {a} [O] {o} [S] {s}",
        "ASO": f"[A] {a} [S] {s} [O] {o}",
        "SAO": f"[S] {s} [A] {a} [O] {o}",
        "OAS": f"[O] {o} [A] {a} [S] {s}",
        "OSA": f"[O] {o} [S] {s} [A] {a}",
    }

def create_aos_sequence_variant(dataset_path):
    """
    Create AOS sequence variants from a dataset with different tag orderings.
    
    Args:
        dataset_path (str): Path to the input CSV dataset
        
    Returns:
        pd.DataFrame: DataFrame with sequence variants for each ordering
    """
    df = pd.read_csv(dataset_path)
    records_with_match = []
    
    for _, row in df.iterrows():
        original_triplet = literal_eval(row['original_triplet'])
        counterfact_triplet = literal_eval(row['counterfact_triplet4_replaced'])
        assert len(original_triplet) == len(counterfact_triplet), "Original and counterfactual triplets must have the same length."
        assert len(original_triplet) > 0, "Original triplet must not be empty."
        assert len(counterfact_triplet) > 0, "Counterfactual triplet must not be empty."

        orig_seq_list = {"AOS": [], "ASO": [], "SAO": [], "OAS": [], "OSA": []}
        cf_seq_list = {"AOS": [], "ASO": [], "SAO": [], "OAS": [], "OSA": []}
        for i, triplet in enumerate(original_triplet):
            orig_a, orig_o, orig_s = triplet
            cf3_a, cf3_o, cf3_s = counterfact_triplet[i]
            orig_seq = create_sequences(orig_a, orig_o, orig_s)
            cf_seq = create_sequences(cf3_a, cf3_o, cf3_s)
            for order, seq in orig_seq.items():
                orig_seq_list[order].append(seq)
                cf_seq_list[order].append(cf_seq[order])

        for order, orig_seqs in orig_seq_list.items():
            final_orig_seq = " [SSEP] ".join(orig_seqs)
            final_cf_seq = " [SSEP] ".join(cf_seq_list[order])
            records_with_match.append({
                "order": order,
                "original_sentence": row["original_sentence"],
                "original_triplet": row["original_triplet"],
                "original_label_variant": final_orig_seq,
                "counterfact4_replaced": row["counterfact4_replaced"],
                "counterfact_triplet4_replaced": row["counterfact_triplet4_replaced"],
                "counterfact_label_variant": final_cf_seq,
                "is_match": row["is_match"]
            })

    return pd.DataFrame(records_with_match)

def format_counterfactuals(input_path):
    """
    Format counterfactual data from CSV into structured format.
    
    Args:
        input_path (str): Path to input CSV file containing counterfactual pairs
        
    Returns:
        pd.DataFrame: Formatted DataFrame with extracted sentences and triplets
    """
    df = pd.read_csv(input_path, encoding="utf-8", quoting=1)

    df['original_sentence'] = df['original_pair'].apply(lambda x: x.split('[A] [O] [S]')[0].strip())
    temp_column = df['original_pair'].apply(lambda x: x.split('[A] [O] [S]')[-1].strip())
    temp_column = temp_column.apply(lambda x: x.split('[SSEP]')).apply(lambda x: [i.strip() for i in x])
    temp_column = temp_column.apply(lambda x: [extract_triplet_fixed(i) for i in x])
    df['original_triplet'] = deepcopy(temp_column)

    try:
        df['counterfact4_replaced'] = df['corrupted_pair'].apply(lambda x: x.split('[A] [O] [S]')[0].strip())
        temp_column = df['corrupted_pair'].apply(lambda x: x.split('[A] [O] [S]')[-1].strip())
        temp_column = temp_column.apply(lambda x: x.split('[SSEP]'))
        temp_column = temp_column.apply(lambda x: [i.strip() for i in x])
        temp_column = temp_column.apply(lambda x: [extract_triplet_fixed(i) for i in x])
        df['counterfact_triplet4_replaced'] = deepcopy(temp_column)
    except KeyError:
        df['counterfact4_replaced'] = None
        df['counterfact_triplet4_replaced'] = None
        print("KeyError: 'corrupted_pair' is not in a valid format (must be string and no None value). Skipping replacement.")

    df_out = df[['index', 'original_sentence', 'original_triplet', 'counterfact4_replaced', 'counterfact_triplet4_replaced']].copy()
    folder = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    print(f"Saving formatted data to {os.path.join(folder, f'formatted_{filename}')}")
    df_out.to_csv(os.path.join(folder, f"formatted_{filename}"), index=False)
    print(f"Saved {len(df_out)} rows to {os.path.join(folder, f'formatted_{filename}')}")
    return df_out


def format_counterfactuals_gas(
    input_path: str,
    col_original: str = "original_pair",
    col_counter: str = "corrupted_pair",
    index_col_name: str = "index",
    coerce_index_to_int: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - index (preserved or generated), optionally coerced to int64 if safe
      - original_sentence (prompt ending with ' =>')
      - original triplet
      - counterfact        (your code keeps ' =>', retained here)
      - counterfact triplet
    Drops rows where counterfact is NaN/empty.
    """
    pair_re = re.compile(r'^(.*?)\s*=>\s*\((.*?)\)\s*$')

    def parse_pair(value: Optional[str]):
        if pd.isna(value):
            return "", ""
        s = str(value).strip()
        m = pair_re.match(s)
        if m:
            left = m.group(1).strip()
            inner = m.group(2).strip()
            return left, f"( {inner} )"
        if "=>" in s:
            left, right = s.split("=>", 1)
            left, right = left.strip(), right.strip()
            if not (right.startswith("(") and right.endswith(")")):
                right = f"( {right} )"
            return left, right
        return "" | ""

    df_in = pd.read_csv(input_path)

    if index_col_name in df_in.columns:
        idx_vals = df_in[index_col_name].copy()
    else:
        idx_vals = pd.Series(df_in.index, name=index_col_name)

    if coerce_index_to_int:
        try:
            idx_num = pd.to_numeric(idx_vals, errors="coerce")
            idx_num = idx_num.replace([np.inf, -np.inf], np.nan)
            if idx_num.notna().all():
                idx_vals = idx_num.astype("int64")
        except Exception:
            pass

    orig_sentences, orig_triplets = [], []
    cf_sentences, cf_triplets = [], []

    for _, row in df_in.iterrows():
        o_s, o_t = parse_pair(row.get(col_original, ""))
        c_s, c_t = parse_pair(row.get(col_counter, ""))

        orig_sentences.append((o_s + " =>").strip())
        orig_triplets.append(o_t)
        cf_sentences.append((c_s + " =>").strip() if c_s else c_s)
        cf_triplets.append(c_t)

    df_out = pd.DataFrame({
        index_col_name: idx_vals,
        "original_sentence": orig_sentences,
        "original_triplet": orig_triplets,
        "counterfact": cf_sentences,
        "counterfact_triplet": cf_triplets,
    })

    mask = df_out["counterfact"].notna() & (df_out["counterfact"].astype(str).str.strip() != "")
    df_out = df_out.loc[mask].reset_index(drop=True)

    folder = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    out_path = os.path.join(folder, f"formatted_{filename}")
    print(f"Saving formatted data to {out_path}")
    df_out.to_csv(out_path, index=False)
    print(f"Saved {len(df_out)} rows to {out_path}")

    return df_out

_GAS_TRIPLET_RE = re.compile(r"\(([^()]*)\)")

def _extract_first_triplet(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _GAS_TRIPLET_RE.search(text)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    # return f"({', '.join(parts)})" # Old spacing with comma. TODO: Remove/clean this part
    return f"( {' | '.join(parts)} )"


def _normalize_triplet_str(s: str) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        return _extract_first_triplet(s)
    return _extract_first_triplet(s)


def filter_correct_data_gas(
    model,
    data: pd.DataFrame,
    sentence_col: str = "original_sentence",
    label_col: str = "original_triplet",
    max_tokens: int = 60,
    filter_only_correct: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    inputs = data[sentence_col].tolist()
    labels = data[label_col].tolist()

    inferences, originals, match_flags = [], [], []

    for prompt, expected_triplet in zip(inputs, labels):
        base_prompt = str(prompt).rstrip()
        if not base_prompt.endswith("=>"):
            base_prompt = base_prompt + " =>"

        output = model.generate(
            input=base_prompt,
            max_new_tokens=max_tokens,
            stop_at_eos=True,
            do_sample=False,
            return_type="str"
        )

        gen_only = output[len(base_prompt):].lstrip() if output.startswith(base_prompt) else output
        gen_triplet_norm = _extract_first_triplet(gen_only)
        exp_triplet_norm = _normalize_triplet_str(expected_triplet)

        is_match = (gen_triplet_norm is not None) and (exp_triplet_norm is not None) and (gen_triplet_norm == exp_triplet_norm)
        if not is_match:
            print(f"Generated triplet: {gen_triplet_norm}, Expected triplet: {exp_triplet_norm}")

        originals.append(exp_triplet_norm if exp_triplet_norm is not None else str(expected_triplet))
        inferences.append(gen_triplet_norm if gen_triplet_norm is not None else "")

        match_flags.append(is_match)

    df_result = data.copy()
    df_result["original_label"] = originals          
    df_result["inference"] = inferences          
    df_result["is_match"] = match_flags

    total = len(df_result)
    correct = int(df_result["is_match"].sum())
    print(f"Correct: {correct} / {total} ({correct / total:.2%})")

    if filter_only_correct:
        df_result = df_result[df_result["is_match"]].reset_index(drop=True)

    if save_path:
        df_result.to_csv(save_path, index=False)

    return df_result


def _normalize_triplet_str(s: str) -> Optional[str]:
    """Return a normalized '(a, b, c)' triplet string or None if not parseable."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    m = _GAS_TRIPLET_RE.search(s if (s.startswith("(") and s.endswith(")")) else f"({s})")
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    # return f"({', '.join(parts)})" # Old spacing with comma. TODO: Remove/clean this part
    return f"( {' | '.join(parts)} )"


def build_eap_dataset_gas(
    model,
    df: pd.DataFrame,
    sentence_col: str = "original_sentence",
    triplet_col: str = "original_triplet",
    corrupted_col: str = "counterfact",
    corrupted_triplet_col: str = "counterfact_triplet",
    filter_same_length_counterfactuals: bool = True,
    append_labels: bool = False,
) -> pd.DataFrame:
    """
    Build an EAP dataset that compares the original prompt vs its counterfactual,
    using full triplet strings as labels. A/O/S modes are ignored.

    Args:
        model: TransformerLens model (used for tokenization).
        df: DataFrame with columns: original_sentence, original triplet,
            counterfact, counterfact triplet.
        sentence_col: original sentence column.
        triplet_col: original triplet column (string like '(a, b, c)').
        corrupted_col: counterfactual sentence column.
        corrupted_triplet_col: counterfactual triplet column.
        filter_same_length_counterfactuals: drop rows where clean/corrupted prompts
            tokenize to different lengths.
        append_labels: (currently unused; placeholder for prefixing label tokens).

    Returns:
        pd.DataFrame with columns: clean, corrupted, correct_label, incorrect_label,
        correct_idx, incorrect_idx.
    """
    eap_rows = []
    num_removed = 0

    for _, row in df.iterrows():
        if not row.get("is_match", True):
            continue
        ctext = row.get(corrupted_col, None)
        if not isinstance(ctext, str) or not ctext.strip():
            continue

        clean = str(row[sentence_col]).rstrip()
        corrupted = ctext.rstrip()

        if filter_same_length_counterfactuals:
            try:
                clean_tokens = model.to_tokens(clean)
                corrupted_tokens = model.to_tokens(corrupted)
                if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
                    print(f"Skipping row due to token length mismatch: {clean_tokens.shape[1]} vs {corrupted_tokens.shape[1]}")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    num_removed += 1
                    continue
            except Exception as e:
                print(f"Error occurred while tokenizing: {e}")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                num_removed += 1
                continue

        correct_label = _normalize_triplet_str(row.get(triplet_col))
        incorrect_label = _normalize_triplet_str(row.get(corrupted_triplet_col))
        if (correct_label is None) or (incorrect_label is None):
            print(f"Skipping row due to unparsable triplet labels (None labels): {row.get(triplet_col)} / {row.get(corrupted_triplet_col)}")
            num_removed += 1
            continue

        try:
            correct_idx = model.to_tokens(" " + correct_label).tolist()
            incorrect_idx = model.to_tokens(" " + incorrect_label).tolist()
        except Exception:
            print(f"Skipping row due to tokenization error for labels: {correct_label} / {incorrect_label}")
            num_removed += 1
            continue

        if len(correct_idx[0]) != len(incorrect_idx[0]):
            print(f"Skipping row due to token length mismatch in labels: {correct_label} ({len(correct_idx[0])}) vs {incorrect_label} ({len(incorrect_idx[0])})")
            num_removed += 1
            continue

        eap_rows.append({
            "clean": clean,
            "corrupted": corrupted,
            "correct_label": correct_label,
            "incorrect_label": incorrect_label,
            "correct_idx": correct_idx,
            "incorrect_idx": incorrect_idx,
        })

    if filter_same_length_counterfactuals:
        print(f"Removed {num_removed} out of {len(df)} rows due to token-length/parse issues.")
    print(f"Filtered data size: {len(eap_rows)}")

    eap_df = pd.DataFrame(eap_rows)
    if not eap_df.empty:
        eap_df["correct_idx"] = eap_df["correct_idx"].apply(str)
        eap_df["incorrect_idx"] = eap_df["incorrect_idx"].apply(str)
    return eap_df