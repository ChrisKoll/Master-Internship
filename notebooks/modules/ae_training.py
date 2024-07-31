"""
Collection of functions used in the model training loop.

This module includes functions for training and testing an Autoencoder model using PyTorch.
"""

# Standard imports
import datetime
from typing import Optional

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm

# Self-built modules
from modules.logging_setup import main_logger

__author__ = "Christian Kolland"
__version__ = 0.1


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    prev_updates: int,
    device: str,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
) -> int:
    """
    Trains the model on the given data.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        prev_updates (int): Number of previous updates.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        writer (torch.utils.tensorboard.SummaryWriter, optional): The TensorBoard writer for logging.

    Returns:
        int: The updated number of training steps.
    """
    # Set the model to training mode
    model.train()

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Put batch to device (cpu or cuda)
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch)
        loss = outputs.loss

        # Backward pass and optimazation
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation

        # Keep track of number of updates
        n_upd = prev_updates + batch_idx
        if n_upd % 100 == 0:
            # Calculate total gradinet norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5  # ?

            main_logger.info(
                f"Step {n_upd:,} (N samples: {n_upd * dataloader.batch_size:,}), Loss: {loss.item():.4f} Grad: {total_norm:.4f}"
            )

            if writer is not None:
                global_step = n_upd
                writer.add_scalar("Loss/Training", loss.item(), global_step)
                writer.add_scalar("GradNorm/Training", total_norm, global_step)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the model parameters
        # Needs to be after gradient norm calculation?
        optimizer.step()

    return prev_updates + len(dataloader)


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cur_step: int,
    device: str,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
) -> None:
    """
    Tests the model on the given data.

    Args:
        model (torch.nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        writer (torch.utils.tensorboard.SummaryWriter, optional): The TensorBoard writer for logging.
    """
    # Set the model to evaluation mode
    model.eval()
    # Track loss
    test_loss = 0

    latent_vectors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # Put batch to device (cpu or cuda)
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)
            latent_vectors.append(outputs.z_sample.cpu().numpy())

            test_loss += outputs.loss.item()

    # Calculate test loss
    test_loss /= len(dataloader)
    main_logger.info(f"===> Test set loss: {test_loss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/Testing", test_loss, global_step=cur_step)

    return np.concatenate(latent_vectors)
