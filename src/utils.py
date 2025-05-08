import torch, re, ast
import pandas as pd
from typing import Optional
from transformers import AutoModelForCausalLM, AutoConfig
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import convert_qwen2_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

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

def build_eap_dataset(
    model: HookedTransformer,
    df: pd.DataFrame,
    sentence_col: str,
    triplet_col: str,
    corrupted_col: str,
    corrupted_triplet_col: str,
    suffix: str,
    idx: int
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

    Returns:
        pd.DataFrame: DataFrame with clean/corrupted prompts, token indices, and raw label texts.
    """
    eap_data = []

    for _, row in df.iterrows():
        clean = row[sentence_col] + f" {suffix}"
        corrupted = row[corrupted_col] + f" {suffix}"
        
        try:
            original_triplet = ast.literal_eval(row[triplet_col])
            corrupted_triplet = ast.literal_eval(row[corrupted_triplet_col])

            correct_label = original_triplet[0][idx]
            incorrect_label = corrupted_triplet[0][idx]

            correct_idx = model.to_tokens(correct_label).tolist()
            incorrect_idx = model.to_tokens(incorrect_label).tolist()

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

    return pd.DataFrame(eap_data)