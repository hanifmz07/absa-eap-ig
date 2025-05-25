import json
import torch
from transformer_lens.train import train
from transformer_lens.train import HookedTransformerTrainConfig
from src.utils import load_model, ABSAAutoRegressiveDataset
from src.utils import apply_active_edge_unfreezing
import argparse
from datetime import datetime
import os
import pickle

def main(args):
    # === Set Device ===
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # === Load ABSA Dataset ===
    with open(args.train_json_path) as f:
        absa_data = json.load(f)

    # === Load Model ===
    model = load_model(args.model_name, device=device)

    # === Load Dataset ===
    dataset = ABSAAutoRegressiveDataset(absa_data, model.tokenizer, shuffle=True, seed=args.seed)

    # === Load Circuit CSV ===
    if not args.train_full_model:
        apply_active_edge_unfreezing(model, args.circuit_csv_path)

    model.train()

    # === Train Config ===
    config = HookedTransformerTrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        print_every=100,
        seed=args.seed,
    )

    # === Set Output Directory ===
    output_dir = args.output_dir
    output_folder_name = f"{datetime.now()}"
    output_folder_name += f'_tflens'
    output_folder_name += f"_{args.train_json_path.split('/')[-1].split('.')[0]}"
    output_folder_name += f'_model-{args.model_name.split("/")[-1]}'
    output_folder_name += f'_lr-{args.lr}'
    output_folder_name += f'_bs-{args.batch_size}'
    output_folder_name += f'_epochs-{args.num_epochs}'
    output_folder_name += f'_{args.circuit_csv_path.split("/")[-1].split(".")[0].split("_")[-1]}' if not args.train_full_model else ''
    output_dir = os.path.join(output_dir, output_folder_name)
    print(f"Output directory: {output_dir}")

    # === Start Training ===
    trained_model = train(model, config, dataset)

    os.makedirs(output_dir, exist_ok=True)
    # === Save Model State ===
    torch.save(trained_model.state_dict(), os.path.join(output_dir, "model.pt"))

    # === Save Model Config ===
    model_cfg_dict = model.cfg.to_dict()
    with open(os.path.join(output_dir, 'model_config.pkl'), 'wb') as f:
        pickle.dump(model_cfg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # === Save Model Tokenizer ===
    model.tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json_path", type=str, required=True, help="Path to the training JSON dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default=f"./results", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--train_full_model", action='store_true', help="Whether to train the full model or not")
    parser.add_argument("--circuit_csv_path", type=str, help="Path to the circuit csv file, required only if --train_full_model is not set (only finetune the circuit)")

    args = parser.parse_args()
    
    if not args.train_full_model and not args.circuit_csv_path:
        parser.error("--circuit_csv_path is required when --train_full_model is not set")

    main(args)