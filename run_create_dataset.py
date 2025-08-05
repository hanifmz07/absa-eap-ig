import argparse
import os
import time
import torch
import pandas as pd
from src import load_finetuned_model_lens_from_dir
from src.utils import (
    filter_correct_data,
    create_full_AOS_dataset,
    create_aos_sequence_variant,
    build_eap_dataset,
    format_counterfactuals
)

def main(args):
    print("Starting create EAP dataset pipeline...")
    
    # === Load Model ===
    print("Loading fine-tuned model...")
    model = load_finetuned_model_lens_from_dir(args.finetuned_model)
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    model.eval()
    print(f"Model loaded and moved to device: {device}")

    # === Step 1: Filter Correct Predictions ===
    print("Reading dataset and filtering correct predictions...")
    df = pd.read_csv(args.dataset_path)

    if "original_pair" in df.columns:
        format_counterfactuals(args.dataset_path)
        folder = os.path.dirname(args.dataset_path)
        filename = os.path.basename(args.dataset_path)
        formated_path = os.path.join(folder, f"formatted_{filename}")
        df = pd.read_csv(formated_path)

    os.makedirs(os.path.dirname(args.filtered_data_path), exist_ok=True)
    filtered_df = filter_correct_data(
        model,
        df,
        "original_sentence",
        "original_triplet",
        filter_mode="AOS",
        filter_only_correct=True,
        save_path=args.filtered_data_path,
        max_tokens=150
    )
    filtered_df.to_csv(args.filtered_data_path, index=False)
    print(f"Filtered data saved to {args.filtered_data_path} ({len(filtered_df)} rows)")

    # === Step 2: Create Full AOS Dataset ===
    df_full = pd.read_csv(args.filtered_data_path)
    full_aos_path = args.full_aos_path

    if "counterfact4_replaced" not in df_full.columns:
        print("Creating full AOS dataset...")
        os.makedirs(os.path.dirname(args.full_aos_path), exist_ok=True)
        full_aos_df = create_full_AOS_dataset(args.filtered_data_path)
        full_aos_df.to_csv(args.full_aos_path, index=False)
        print(f"Full AOS dataset saved to {args.full_aos_path} ({len(full_aos_df)} rows)")

    else:
        full_aos_path = args.filtered_data_path

    # === Step 3: Create AOS Sequence Variants ===
    print("Creating AOS sequence variants...")
    os.makedirs(os.path.dirname(args.sequence_variants_path), exist_ok=True)
    sequence_df = create_aos_sequence_variant(full_aos_path)
    sequence_df.to_csv(args.sequence_variants_path, index=False)
    print(f"AOS sequence variants saved to {args.sequence_variants_path} ({len(sequence_df)} rows)")

    # === Step 4: Build EAP Dataset ===
    print("Building EAP dataset...")
    os.makedirs(os.path.dirname(args.eap_output_path), exist_ok=True)
    eap_df = build_eap_dataset(
        model=model,
        df=sequence_df,
        sentence_col="original_sentence",
        triplet_col="original_label_variant",
        corrupted_col="counterfact4_replaced",
        corrupted_triplet_col="counterfact_label_variant",
        filer_same_length_counterfactuals=True,
        suffix="[A] [O] [S]"
    )
    eap_df.to_csv(args.eap_output_path, index=False)
    print(f"EAP dataset saved to {args.eap_output_path} ({len(eap_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--filtered_data_path", type=str, required=True)
    parser.add_argument("--full_aos_path", type=str, required=True)
    parser.add_argument("--sequence_variants_path", type=str, required=True)
    parser.add_argument("--eap_output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
