from typing import Optional, Literal
import json
import numpy as np
import random
import torch
from src.utils import load_model, ABSAAutoRegressiveDataset, apply_active_edge_unfreezing
from transformer_lens.train import train, HookedTransformerTrainConfig
from transformer_lens import HookedTransformer
from franken_adapter.utils import SundaEmbeddingDataset, load_sundanese_paragraphs

def instruction_tuning(
    model_name: str,
    dataset_path: str,
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    save_path: Optional[str] = None,
    circuit_path: Optional[str] = None,
    finetune_model: Optional[str] = None,
    seed: int = 42
) -> HookedTransformer:
    # === Set global seed ===
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # === Load model ===
    if finetune_model:
        print(f"Loading base model '{model_name}' with fine-tuned weights from: {finetune_model}")
    else:
        print(f"Loading model: {model_name}")
    model = load_model(model_name, device=device, fine_tuned_model_path=finetune_model)

    # === Load dataset ===
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path) as f:
        absa_data = json.load(f)
    dataset = ABSAAutoRegressiveDataset(absa_data, model.tokenizer, seed=seed)

    # === Apply selective unfreezing if needed ===
    if circuit_path:
        print(f"Applying selective unfreezing from circuit file: {circuit_path}")
        apply_active_edge_unfreezing(model, circuit_path)
    else:
        print("No circuit path provided; using default parameter setup.")

    # === Freeze embeddings ===
    print("Freezing embedding layer (W_E)")
    model.embed.W_E.requires_grad = False

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


def embedding_training(
    model_name: str,
    jsonl_path: str,
    save_path: Optional[str] = None,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-4,
    max_len: int = 128,
):
    """
    Trains only the embedding layer of a transformer model using Sundanese wiki data.

    Args:
        model_name (str): HF model name to load.
        jsonl_path (str): Path to the filtered wiki JSONL file.
        save_path (Optional[str]): Where to save the final model weights, or None to skip saving.
        device (str): Device to train on.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        max_len (int): Max token length.
    """
    print("Loading Sundanese wiki data...")
    sentences = load_sundanese_paragraphs(jsonl_path)
    print(f"Loaded {len(sentences)} filtered paragraphs.")

    print("Loading base model...")
    model = load_model(model_name, device=device)

    print("Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False

    print("Unfreezing embedding layer...")
    model.embed.W_E.requires_grad = True

    dataset = SundaEmbeddingDataset(
        text_list=sentences,
        tokenizer=model.tokenizer,
        max_len=max_len,
        shuffle=True,
    )

    config = HookedTransformerTrainConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        print_every=10,
        save_every=None,
        save_dir=None,
    )

    print("Starting training (embedding only)...")
    model.train()
    trained_model = train(model, config, dataset)
    print("Training completed.")

    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(trained_model.state_dict(), save_path)
    else:
        print("No save_path provided — model was not saved.")

    return trained_model


def merge_instruction_with_embedding(
    base_model_name: str,
    instruct_weights_path: str,
    embedding_weights_path: str,
    output_path: str,
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    ) -> HookedTransformer:
    
    """
    Loads two models with different weights, swaps the embedding layer,
    saves the resulting Frankenstein model, and returns it.

    Args:
        base_model_name (str): Name of the base HuggingFace model.
        instruct_weights_path (str): Path to the instruction-tuned weights (Step 2a).
        embedding_weights_path (str): Path to the embedding-tuned weights (Step 2b).
        output_path (str): Path to save the final merged model.
        device (str): Device to load the models on ("cuda" or "cpu").

    Returns:
        HookedTransformer: The merged FrankenAdapter model.
    """
    
    print("Loading instruction-tuned model...")
    instruct_model = HookedTransformer.from_pretrained(base_model_name, device=device)
    state_dict_instruct = torch.load(instruct_weights_path, map_location=device)
    instruct_model.load_state_dict(state_dict_instruct)

    print("Loading embedding-tuned model...")
    model_embedding = HookedTransformer.from_pretrained(base_model_name, device=device)
    state_dict_embedding = torch.load(embedding_weights_path, map_location=device)
    model_embedding.load_state_dict(state_dict_embedding)

    print("Swapping embeddings...")
    instruct_model.embed.W_E.data = model_embedding.embed.W_E.data.clone()

    print(f"Saving merged model to: {output_path}")
    torch.save(instruct_model.state_dict(), output_path)

    return instruct_model


def finetune_franken_adapter(
    base_model_name: str,
    franken_weights_path: str,
    json_data_path: str,
    device: Literal["cuda", "cpu", "mps"] = "cuda",
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-4,
    freeze_embedding: bool = True,
    circuit_path: Optional[str] = None,
    save_path: Optional[str] = None,
    seed: int = 42
) -> HookedTransformer:
    """
    Fine-tunes a FrankenAdapter model on a new dataset (e.g., Sundanese ABSA).
    
    Args:
        base_model_name (str): Name of the pretrained model (e.g., "Qwen/Qwen2.5-0.5B").
        franken_weights_path (str): Path to the FrankenAdapter weights.
        json_data_path (str): Path to the fine-tuning JSON dataset.
        device (str): Device to use ("cuda" or "cpu").
        num_epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Learning rate.
        freeze_embedding (bool): Whether to freeze the embedding layer (`W_E`).
        save_path (Optional[str]): Optional path to save the fine-tuned model.
        seed (int): Random seed for reproducibility.
    
    Returns:
        HookedTransformer: The fine-tuned FrankenAdapter model.
    """

    # === Set global seed ===
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    print("Loading FrankenAdapter model...")
    model = HookedTransformer.from_pretrained(base_model_name, device=device)
    state_dict = torch.load(franken_weights_path, map_location=device)
    model.load_state_dict(state_dict)

    print(f"Loading dataset from {json_data_path}")
    with open(json_data_path) as f:
        data = json.load(f)
    dataset = ABSAAutoRegressiveDataset(data, model.tokenizer, seed=seed)

    if circuit_path:
        print(f"Applying selective unfreezing from circuit file: {circuit_path}")
        apply_active_edge_unfreezing(model, circuit_path)
    else:
        print("No circuit path provided; using default parameter setup.")

    if freeze_embedding:
        print("Freezing embedding layer (W_E)")
        model.embed.W_E.requires_grad = False
    else:
        print("Keeping embedding layer trainable")

    config = HookedTransformerTrainConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        print_every=10,
        save_every=None,
        save_dir=None,
    )

    print("Starting fine-tuning...")
    model.train()
    trained_model = train(model, config, dataset)
    print("Fine-tuning complete.")

    if save_path:
        print(f"Saving fine-tuned model to: {save_path}")
        torch.save(trained_model.state_dict(), save_path)

    return trained_model
