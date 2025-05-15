import torch, re, ast
import pandas as pd
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights, convert_bloom_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from torch.utils.data import Dataset, DataLoader
from eap.graph import Graph

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
    return " " + " ".join([mapping[c] for c in mode if c in mapping])


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
