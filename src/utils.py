import torch, re, ast
import pandas as pd
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from torch.utils.data import Dataset, DataLoader

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
    raise ValueError(f"Unknown or unsupported model type for: {base_model_name}")


def load_finetuned_model(base_model_name: str, fine_tuned_model_path: str) -> HookedTransformer:
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
    state_dict = convert_qwen2_weights(hf_model, cfg)
    model = HookedTransformer.from_pretrained(
        model_name=base_model_name,
        device=device
    )
    model.load_and_process_state_dict(state_dict)

    return model


def convert_triplet_string(triplet_str: str) -> str:
    """
    Convert a triplet string to formatted label text.

    Args:
        triplet_str (str): A string representation of a triplet.

    Returns:
        str: Formatted triplet as "[A] aspect [O] opinion [S] sentiment"
    """
    try:
        triplet = ast.literal_eval(triplet_str)[0]
        aspect, opinion, sentiment = triplet
        return f"[A] {aspect} [O] {opinion} [S] {sentiment}"
    except (ValueError, SyntaxError, IndexError):
        return ""
    

def filter_correct_data(
    model,
    data: pd.DataFrame,
    sentence_col: str,
    label_col: str,
    max_tokens: int = 50,
    suffix: str = " [A] [O] [S]",
    filter_only_correct: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Filters out samples where the model's generated output does not match the expected label,
    and appends inference results back into the original DataFrame.

    Args:
        model: TransformerLens HookedTransformer model.
        data (pd.DataFrame): Original DataFrame with input and label columns.
        sentence_col (str): Column containing input sentences.
        label_col (str): Column containing expected triplets.
        max_tokens (int): Max tokens to generate.
        suffix (str): Prompt suffix to guide generation.
        filter_only_correct (bool): If True, keep only correct rows.
        save_path (Optional[str]): If given, save the output CSV here.

    Returns:
        pd.DataFrame: The original DataFrame with added columns:
                      'original_label', 'inference', and 'is_match'.
    """
    inputs = data[sentence_col].tolist()
    labels = data[label_col].tolist()

    results, tags, exp_labels = [], [], []

    for prompt, label_raw in zip(inputs, labels):
        full_prompt = prompt + suffix
        output = model.generate(
            input=full_prompt,
            max_new_tokens=max_tokens,
            stop_at_eos=True,
            do_sample=False,
            return_type="str"
        )

        cleaned = re.sub(r"<\|endoftext\|>", "", output)

        if "[SSEP]" in cleaned:
            cleaned = cleaned.split("[SSEP]", 1)[-1].strip()
        elif cleaned.count("[A]") > 1:
            cleaned = cleaned.split("[A] [O] [S]", 1)[-1].strip()
        else:
            cleaned = cleaned.split(".", 1)[-1].strip()

        expected = convert_triplet_string(label_raw)
        is_match = cleaned == expected

        exp_labels.append(expected)
        results.append(cleaned)
        tags.append(is_match)

    # Add columns back to the original dataframe
    df_result = data.copy()
    df_result["original_label"] = exp_labels
    df_result["inference"] = results
    df_result["is_match"] = tags

    # Print stats
    total = len(df_result)
    correct = df_result["is_match"].sum()
    print(f"✅ Correct: {correct} / {total} ({correct / total:.2%})")

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
    idx: int,
    filer_same_length_counterfactuals: bool = True,
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
        idx (int): Index in the triplet to extract (0=aspect, 1=opinion, 2=sentiment).
        filer_same_length_counterfactuals (bool): If True, remove datapoints where
            counterfactual token length differs from original token length.

    Returns:
        pd.DataFrame: DataFrame with clean/corrupted prompts, token indices, and raw label texts.
    """
    eap_data = []
    num_removed = 0
    for _, row in df.iterrows():
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

            # Peter: need to be careful here with the extra space in front of the string label,
            # i.e if model is prediction "... [A]", then you need to prefix a space before
            # label so correct label is " sushi" instead of "sushi". Not sure
            # when this does not apply though, need to re-access when we do multi-task
            # prediction like "[A] sushi [O]"
            correct_idx = model.to_tokens(f" {correct_label}").tolist()
            incorrect_idx = model.to_tokens(f" {incorrect_label}").tolist()

            if len(correct_idx[0]) != len(incorrect_idx[0]):
                num_removed += 1
                continue

            # if label has multiple tokens, append all but last label tokens to clean and corrupted
            # because we need to obtain the right logit conditioned on the correct prefix
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

