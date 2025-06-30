import argparse
import os
from datetime import datetime
from franken_adapter.franken import instruction_tuning

def main():
    parser = argparse.ArgumentParser(description="Run instruction tuning with Franken Adapter.")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model name or path.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the training JSON dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs/models/franken", help="Base directory to save output.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--circuit_path", type=str, default=None, help="Optional circuit CSV path.")
    parser.add_argument("--finetune_model", type=str, default=None, help="Optional path to a fine-tuned model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # === Generate output filename ===
    output_file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tflens"
    output_file_name += f"_{os.path.basename(args.json_path).split('.')[0]}"
    output_file_name += f'_model-{args.model_name.split("/")[-1]}'
    output_file_name += f'_lr-{args.lr}'
    output_file_name += f'_bs-{args.batch_size}'
    output_file_name += f'_epochs-{args.num_epochs}'
    if args.circuit_path:
        output_file_name += f'_{os.path.basename(args.circuit_path).split(".")[0].split("_")[-1]}'

    os.makedirs(args.output_dir, exist_ok=True)

    output_file_name += ".pt"
    
    final_output_dir = os.path.join(args.output_dir, output_file_name)
    print(f"Output directory: {final_output_dir}")

    instruction_tuning(
        model_name=args.model_name,
        dataset_path=args.json_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=final_output_dir,
        circuit_path=args.circuit_path,
        finetune_model=args.finetune_model,
        seed=args.seed
    )

if __name__ == "__main__":
    main()