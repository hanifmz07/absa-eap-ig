# Modified from transformer_lens training script to add save based on based on best loss

from dataclasses import dataclass
from typing import Any, Optional, Literal

import torch
import torch.optim as optim
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformer_lens import utils
from transformer_lens.HookedTransformer import HookedTransformer

import pickle
import os

@dataclass
class HookedTransformerTrainConfig:
    """
    Configuration class to store training hyperparameters for a training run of
    an HookedTransformer model.
    Args:
        num_epochs (int): Number of epochs to train for
        batch_size (int): Size of batches to use for training
        lr (float): Learning rate to use for training
        seed (int): Random seed to use for training
        momentum (float): Momentum to use for training
        max_grad_norm (float, *optional*): Maximum gradient norm to use for
        weight_decay (float, *optional*): Weight decay to use for training
        optimizer_name (str): The name of the optimizer to use
        device (str, *optional*): Device to use for training
        warmup_steps (int, *optional*): Number of warmup steps to use for training
        save_mode (Literal["best", "steps"], *optional*): Save mode - "best" saves best model, "steps" saves every N steps
        save_every (int, *optional*): After how many batches should a checkpoint be saved (used with save_mode="steps")
        save_dir, (str, *optional*): Where to save checkpoints
        wandb (bool): Whether to use Weights and Biases for logging
        wandb_project (str, *optional*): Name of the Weights and Biases project to use
        print_every (int, *optional*): Print the loss every n steps
        max_steps (int, *optional*): Terminate the epoch after this many steps. Used for debugging.
    """

    num_epochs: int
    batch_size: int
    val_batch_size: Optional[int] = None
    lr: float = 1e-4
    seed: int = 0
    momentum: float = 0.0
    max_grad_norm: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_name: str = "Adam"
    device: Optional[str] = None
    warmup_steps: int = 0
    save_mode: Optional[Literal["best", "steps", None]] = None
    save_every: Optional[int] = None
    save_dir: Optional[str] = None
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    print_every: Optional[int] = 50
    max_steps: Optional[int] = None
    validation_mode: Optional[Literal["epoch", "steps", None]] = None
    validation_steps: Optional[int] = None
    top_k: Any = None
    sample_size: Any = None
    subtract_data_amount: int = 0

def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
) -> HookedTransformer:
    """
    Trains an HookedTransformer model on an autoregressive language modeling task.
    Args:
        model: The model to train
        config: The training configuration
        dataset: The dataset to train on - this function assumes the dataset is set up for autoregressive language modeling.
    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)
    model.train()
    if config.wandb:
        if config.wandb_project is None:
            config.wandb_project = "easy-transformer"
        wandb.init(project=config.wandb_project, config=vars(config), name=config.wandb_run_name)

    if config.device is None:
        config.device = utils.get_device()

    optimizer: Optimizer
    if config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(config.weight_decay if config.weight_decay is not None else 0.0),
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model.to(config.device)

    # Initialize best loss for "best" save mode
    best_loss = float("inf")

    if config.save_mode is not None and config.save_dir is None:
        raise ValueError("save_dir must be specified if save_mode is used")

    # Initialize save folder
    if config.save_mode is not None and config.save_dir is not None:
        os.makedirs(config.save_dir, exist_ok=True)

        # Save model config
        model_cfg_dict = model.cfg.to_dict()
        with open(os.path.join(config.save_dir, "model_config.pkl"), "wb") as f:
            pickle.dump(model_cfg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save tokenizer
        model.tokenizer.save_pretrained(config.save_dir)

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        samples = 0
        epoch_loss = 0.0
        for step, batch in tqdm(enumerate(dataloader)):
            tokens = batch["tokens"].to(config.device)
            loss = model(tokens, return_type="loss")
            loss.backward()
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]

            # Add loss to epoch loss
            epoch_loss += loss.item()

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch})

            if config.print_every is not None and step % config.print_every == 0:
                print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            # Handle saving based on mode
            if config.save_dir is not None:
                if config.save_mode == "every" and config.save_every is not None:
                    # Save every N steps
                    if step % config.save_every == 0:
                        torch.save(model.state_dict(), f"{config.save_dir}/model_epoch{epoch}_step{step}.pt")
            
            # Handle validation based on "steps" mode
            if val_dataset is not None and config.validation_mode == "steps" and config.validation_steps:
                if step % config.validation_steps == 0 and step > 0:
                    print(f"Running validation at step {step} of epoch {epoch}...")
                    val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)
                    model.eval()
                    val_loss = 0.0
                    val_samples = 0
                    with torch.no_grad():
                        for val_step, val_batch in tqdm(enumerate(val_dataloader)):
                            val_tokens = val_batch["tokens"].to(config.device)
                            loss = model(val_tokens, return_type="loss")
                            val_loss += loss.item()
                            val_samples += val_tokens.shape[0]
                    avg_val_loss = val_loss / len(val_dataloader)
                    print(f"Validation loss at step {step} of epoch {epoch}: {avg_val_loss:.4f}")
                    if config.wandb:
                        wandb.log({"val_loss": avg_val_loss, "epoch": epoch, "step": step})
                    model.train()

            # Handle max steps
            if config.max_steps is not None and step >= config.max_steps:
                break

        # End of epoch - print average loss
        print(f"Average loss for epoch {epoch}: {epoch_loss / (step + 1):.4f}")
        
        # End of epoch - handle "best" save mode
        if config.save_mode == "best":
            # Calculate average epoch loss
            epoch_loss /= (step + 1)
            # if loss.item() < best_loss:
            if epoch_loss < best_loss:
                # best_loss = loss.item()
                best_loss = epoch_loss

                # Save the model
                torch.save(model.state_dict(), f"{config.save_dir}/model.pt")
                print(f"New best loss on epoch {epoch}: {best_loss:.4f} - Model saved!")
        
        # Optional: Validate on validation dataset
        # print(f"Validation mode: {config.validation_mode}")
        # print(f"Val dataset is {'not ' if val_dataset is None else ''}None")
        # print(f"Condition 1: {config.validation_mode == 'epoch'}")
        # print(f"Condition 2: {val_dataset is not None}")
        if val_dataset is not None and config.validation_mode == "epoch":
            print(f"Running validation in the end of an epoch {epoch}...")
            val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False)
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for val_step, val_batch in tqdm(enumerate(val_dataloader)):
                    val_tokens = val_batch["tokens"].to(config.device)
                    loss = model(val_tokens, return_type="loss")
                    val_loss += loss.item()
                    val_samples += val_tokens.shape[0]
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation loss after epoch {epoch}: {avg_val_loss:.4f}")
            if config.wandb:
                wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
            model.train()


    return model