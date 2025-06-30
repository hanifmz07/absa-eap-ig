import argparse
from typing import Optional
from franken_adapter.franken import finetune_franken_adapter 

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a FrankenAdapter model on a new dataset.")

    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base pretrained model.")
    parser.add_argument("--franken_weights_path", type=str, required=True, help="Path to FrankenAdapter weights (.pt).")
    parser.add_argument("--json_data_path", type=str, required=True, help="Path to training data JSON file.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default="cuda", help="Device to train on.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--freeze_embedding", action="store_true", help="Freeze embedding layer W_E.")
    parser.add_argument("--circuit_path", type=str, default=None, help="Path to circuit CSV file for selective unfreezing.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save fine-tuned model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    finetune_franken_adapter(
        base_model_name=args.base_model_name,
        franken_weights_path=args.franken_weights_path,
        json_data_path=args.json_data_path,
        device=args.device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_embedding=args.freeze_embedding,
        circuit_path=args.circuit_path,
        save_path=args.save_path,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
