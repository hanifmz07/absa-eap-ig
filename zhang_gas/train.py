import torch
import random
import numpy as np
from typing import Optional, Literal
from transformer_lens import HookedTransformer
from transformer_lens.train import train, HookedTransformerTrainConfig
from src.utils import load_model
from zhang_gas.data import open_data, preprocess_data

def zhang_gas_train(
    model_name: str,
    dataset_path: str,
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    finetune_model: Optional[str] = None,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> HookedTransformer:
    # === Set global seed ===
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # === Load model ===
    model = load_model(model_name, device=device, fine_tuned_model_path=finetune_model)

    # === Load dataset ===
    print(f"Loading dataset from: {dataset_path}")
    # dataset = ABSAAutoRegressiveDataset(absa_data, model.tokenizer, seed=seed)
    dataset = open_data(json_path=dataset_path)
    dataset = preprocess_data(dataset, model.tokenizer)

    # === Training config ===
    config = HookedTransformerTrainConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        print_every=10,
        save_every=None,
        save_dir=None
    )

    print("Starting training...")
    model.train()
    trained_model = train(model, config, dataset)
    print("Training completed.")

    # === Save model ===
    if save_path:
        print(f"Saving model to: {save_path}")
        torch.save(trained_model.state_dict(), save_path)
    else:
        print("No save path provided; model will not be saved.")

    return trained_model