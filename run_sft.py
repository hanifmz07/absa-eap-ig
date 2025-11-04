import json
import os
import pickle
import random
import argparse
from datetime import datetime
from time import time

import torch
from src.train import HookedTransformerTrainConfig, train

from src.utils import load_model, ABSAAutoRegressiveDataset
from src.utils import apply_active_edge_unfreezing

from dotenv import load_dotenv
load_dotenv()

import wandb

# --- Sampling / difficulty tools ---
from src.sampling import (
    # MVP
    add_element_order_to_json,
    annotate_difficulty,
    stratified_sample_by_factors,
    # GAS
    add_element_order_to_json_gas,
    annotate_difficulty_gas,
    stratified_sample_by_factors_gas,
)

def main(args):
    # === Set Device ===
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # === Prepare (maybe-sampled) training data ===
    absa_data = None

    if args.sample_size:
        if args.prompt_type == "gas":
            # -------- GAS pipeline --------
            src_json = args.inference_train_json_path
            out_dir = os.path.dirname(src_json)

            # 0) ensure 'aos' tag
            add_element_order_to_json_gas(src_json)

            # 1) annotate with metrics + difficulty
            annotated_path = os.path.join(out_dir, str(args.sample_size)+"_gas_full_annotated.json")
            annotated, thresholds = annotate_difficulty_gas(
                input_path=src_json,
                output_path=annotated_path,
                difficulty_method="quantile",
                bins=3,
                min_easy_f1=1.0,
            )
            print("GAS difficulty thresholds:", thresholds)

            # 2) stratified sample (instances, not sentences)
            sample_path = os.path.join(out_dir, str(args.sample_size)+"_gas_sample.json")
            sample, strata = stratified_sample_by_factors_gas(
                full_data_path=annotated_path,
                output_path=sample_path,
                factors=["triplet_count", "input_length_bin", "difficulty"],
                target_total_samples=args.sample_size,
                permutations_per_sentence=1,
                seed=args.seed,
                plot=False,
            )
            print(f"GAS sampled {len(sample)} instances → {sample_path}")

            # 3) load sampled as training data
            with open(sample_path, "r", encoding="utf-8") as f:
                absa_data = json.load(f)

        else:
            # -------- MVP pipeline (AOS-style) --------
            # Use the main training JSON as the source
            src_json = args.inference_train_json_path
            out_dir = os.path.dirname(src_json)

            # 0) parse and set element_order from input suffix (e.g., [A][O][S])
            add_element_order_to_json(src_json)

            # 1) annotate with metrics + difficulty (task inferred from element_order)
            annotated_path = os.path.join(out_dir, str(args.sample_size)+"_mvp_full_annotated.json")
            annotated, thresholds = annotate_difficulty(
                input_path=src_json,
                output_path=annotated_path,
                difficulty_method="quantile",
                bins=3,
                min_easy_f1=1.0,
            )
            print("MVP difficulty thresholds:", thresholds)

            # 2) stratified sample
            sample_path = os.path.join(out_dir, str(args.sample_size)+"_mvp_sample.json")
            sample, strata = stratified_sample_by_factors(
                full_data_path=annotated_path,
                output_path=sample_path,
                factors=["triplet_count", "input_length_bin", "difficulty"],
                target_total_samples=args.sample_size,
                # MVP often expands permutations later; keep 5 if your pipeline expects that,
                # otherwise set to 1 to treat each item as a single instance.
                permutations_per_sentence=5,
                seed=args.seed,
                plot=False,
            )
            print(f"MVP sampled {len(sample)} (sentence-permuted instances) → {sample_path}")

            with open(sample_path, "r", encoding="utf-8") as f:
                absa_data = json.load(f)
    else:
        # no sampling → just read the training JSON
        with open(args.train_json_path, "r", encoding="utf-8") as f:
            absa_data = json.load(f)

    # === Load Model ===
    model = load_model(args.model_name, device=device)

    # === Load Dataset ===
    dataset = ABSAAutoRegressiveDataset(
        absa_data,
        model.tokenizer,
        shuffle=True,
        seed=args.seed,
        max_len=300,
    )

    val_dataset = None
    if args.val_json_path:
        print(f"Loading validation dataset from {args.val_json_path}...")
        with open(args.val_json_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        val_dataset = ABSAAutoRegressiveDataset(
            val_data,
            model.tokenizer,
            shuffle=False,
            seed=args.seed,
            max_len=300,
        )

    # === Circuit-unfreezing (targeted finetune) ===
    if not args.train_full_model:
        model = apply_active_edge_unfreezing(model, args.circuit_csv_path)

    model.train()

    # === Set Output Directory ===
    output_dir = args.output_dir
    output_folder_name = f"{datetime.now()}"
    output_folder_name += "_tflens"
    output_folder_name += f"_{os.path.splitext(os.path.basename(args.train_json_path))[0]}"
    output_folder_name += f"_model-{args.model_name.split('/')[-1]}"
    output_folder_name += f"_lr-{args.lr}"
    output_folder_name += f"_bs-{args.batch_size}"
    output_folder_name += f"_epochs-{args.num_epochs}"
    output_folder_name += (
        f"_{os.path.splitext(os.path.basename(args.circuit_csv_path))[0].split('_')[-1]}"
        if not args.train_full_model
        else ""
    )
    if args.sample_size is not None:
        output_folder_name += f"_n{args.sample_size}_{args.prompt_type}"

    output_dir = os.path.join(output_dir, output_folder_name)
    print(f"Output directory: {output_dir}")

    # Print all configurations
    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Train JSON: {args.train_json_path}")
    print(f"Inference Train JSON: {args.inference_train_json_path}")
    print(f"Val JSON: {args.val_json_path}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output dir: {output_dir}")
    print(f"Circuit CSV: {args.circuit_csv_path}")
    print(f"Train full model: {args.train_full_model}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Val batch size: {args.val_batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Save mode: {args.save_mode}")
    print(f"Validation mode: {args.validation_mode}")
    print(f"Val dataset: {'Yes' if val_dataset else 'No'}")
    print(f"Device: {device}")
    print("=" * 50)

    if not args.train_full_model:
        top_k = os.path.basename(args.circuit_csv_path).split('_')[-1].replace('.csv','')
    else:
        top_k = "fullsft"
    wandb_run_name = f"seed-{args.seed}_{top_k}_optimizer-{args.optimizer}_lr-{args.lr}_samplesize-{args.sample_size}_data-{args.train_json_path.split('/')[2]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # === Train Config (AdamW + WD 1e-2) ===
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = HookedTransformerTrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        print_every=1,
        seed=args.seed,
        save_mode=args.save_mode,
        save_dir=output_dir,
        validation_mode="epoch" if val_dataset else None,
        val_batch_size=args.val_batch_size,
        wandb=True,
        wandb_project="absa-eap",
        wandb_run_name=wandb_run_name,
        optimizer_name=args.optimizer,
        weight_decay=1e-2 if args.optimizer == "AdamW" else None,
    )

    # For throughput stats, use the actual count
    sample_size = len(absa_data)

    # === Start Training ===
    start = time()
    trained_model = train(model, config, dataset, val_dataset=val_dataset)
    elapsed = time() - start

    total_samples = sample_size * args.num_epochs
    samples_per_sec = total_samples / elapsed if elapsed > 0 else float("inf")

    total_steps = args.num_epochs * (sample_size // args.batch_size)
    steps_per_sec = total_steps / elapsed if elapsed > 0 else float("inf")

    print(
        f"{sample_size} samples processed in {elapsed:.2f}s with {args.num_epochs} epochs: "
        f"({samples_per_sec:.2f} samples/sec)"
    )
    print(f"{total_steps} steps in {elapsed:.2f}s ({steps_per_sec:.2f} steps/sec)")
    print("=======================================\n\n")

    if args.save_mode is None:
        print("Saving model on the last epoch...")
        os.makedirs(output_dir, exist_ok=True)
        # === Save Model State ===
        torch.save(trained_model.state_dict(), os.path.join(output_dir, "model.pt"))

        # === Save Model Config ===
        model_cfg_dict = model.cfg.to_dict()
        with open(os.path.join(output_dir, "model_config.pkl"), "wb") as f:
            pickle.dump(model_cfg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # === Save Tokenizer ===
        model.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json_path", type=str, required=True, help="Path to the training JSON dataset")
    parser.add_argument("--inference_train_json_path", type=str, required=False, help="Path to JSON used for GAS sampling (if prompt_type=gas)")
    parser.add_argument("--prompt_type", type=str, choices=["gas", "mvp"], default="gas", help="Sampling/prompt style")
    parser.add_argument("--save_mode", type=str, choices=["best", "every", None], default=None, help="Model saving mode during training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")

    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adam", "SGD"], default="Adam", help="Optimizer to use")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
    parser.add_argument("--val_json_path", type=str, required=False, help="Path to the validation JSON dataset")
    parser.add_argument("--validation_mode", type=str, choices=["epoch", "steps", None], default="epoch", help="Validation mode during training")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Batch size for validation")
    
    parser.add_argument("--train_full_model", action="store_true", help="Whether to train the full model or only the circuit")
    parser.add_argument("--circuit_csv_path", type=str, help="Path to the circuit csv file (required if NOT --train_full_model)")
    parser.add_argument("--sample_size", type=int, default=None, help="Target number of training instances after sampling (None = use all)")

    args = parser.parse_args()

    if not args.train_full_model and not args.circuit_csv_path:
        parser.error("--circuit_csv_path is required when --train_full_model is not set")
    if args.sample_size and args.prompt_type == "gas" and not args.inference_train_json_path:
        parser.error("--inference_train_json_path is required when --sample_size is set and --prompt_type=gas")

    main(args)
