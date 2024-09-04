"""
Module for training and evaluating an autoencoder model with outer cross-validation.

This module provides functionality to fit an autoencoder model to single-cell data using 
outer cross-validation. It includes functions for training, validating, and testing the model, 
with support for TensorBoard logging and optional progress logging.

Functions:
- fit: Sets up the training process, including the optimizer and TensorBoard writer, and 
  calls the training function for each fold.
- train_fold: Trains the model on specified folds, including data splitting, epoch-wise 
  training, validation, and testing.
- train_epoch: Performs a single training epoch, updating model parameters and logging 
  metrics.
- val_epoch: Validates the model for one epoch, computing and logging the validation loss.
- test_fold: Tests the model on a specific fold, computing and logging the test loss, and 
  generating performance plots.
"""

# Standard imports
from datetime import datetime
from logging import Logger
from typing import Optional

# Third-party imports
from anndata import AnnData
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Self-built modules
from src.autoencoder.ae_model import Autoencoder
import src.utils.data_utils as dutils
from src.utils.json_utils import OptimizerConfig
from src.utils.logging_utils import log_train_metrics, log_val_metrics

__author__ = "Christian Kolland"
__version__ = 1.0


def fit(
    model: Autoencoder,
    model_name: str,
    adata: AnnData,
    data_layer: str,
    folds: list[str],
    optim: OptimizerConfig,
    batch_size: int = 128,
    num_epochs: int = 50,
    device: str = "cpu",
    logger: Optional[Logger] = None,
) -> None:
    """
    Fits the autoencoder model to the provided data.

    This function sets up the training process, including the optimizer and
    TensorBoard writer, and then calls the training function for each fold.

    Args:
        model: The autoencoder model to be trained.
        model_name: A string identifier for the model.
        adata: An AnnData object containing the single-cell data.
        data_layer: The key in adata.layers where the input data is stored.
        optim: An OptimizerConfig object specifying the optimizer settings.
        batch_size: The batch size for training. Defaults to 128.
        num_epochs: The number of epochs to train for. Defaults to 50.
        device: The device to run the training on ("cpu" or "cuda"). Defaults to "cpu".
        logger: An optional Logger object for logging output.

    Returns:
        None
    """
    # Assemble optimizer
    learning_rate = optim.learning_rate
    weight_decay = optim.weight_decay
    optimizer = optim.optimizer(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    writer = SummaryWriter(
        f'runs/hca/{model_name}_{datetime.now().strftime("%d%m%Y-%H%M")}'
    )

    # List of folds
    # donors = ["D1", "H2", "D5"]

    train_fold(
        model,
        folds,
        adata,
        data_layer,
        optimizer,
        batch_size,
        num_epochs,
        device,
        writer,
        logger,
    )

    writer.close()


def train_fold(
    model: Autoencoder,
    folds: list[str],
    adata: AnnData,
    data_layer: str,
    optimizer: optim.Optimizer,
    batch_size: int = 128,
    num_epochs: int = 50,
    device: str = "cpu",
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
) -> None:
    """
    Trains the model on the specified folds of data.

    This function performs the actual training process, including data splitting,
    epoch-wise training, validation, and testing.

    Args:
        model: The autoencoder model to be trained.
        folds: A list of fold identifiers (e.g., donor IDs).
        adata: An AnnData object containing the single-cell data.
        data_layer: The key in adata.layers where the input data is stored.
        optimizer: The optimizer to use for training.
        batch_size: The batch size for training. Defaults to 128.
        num_epochs: The number of epochs to train for. Defaults to 50.
        device: The device to run the training on ("cpu" or "cuda"). Defaults to "cpu".
        writer: An optional SummaryWriter object for TensorBoard logging.
        logger: An optional Logger object for logging output.

    Returns:
        None
    """
    for fold_idx, fold in enumerate(folds):
        if logger is not None:
            logger.info(f">>> FOLD {fold_idx + 1}/{len(folds)} - {fold}")

        # Create outer fold -> Testing
        # Test data from this donor will not be seen during training
        train_data, test_loader = dutils.create_outer_fold(
            adata, fold, data_layer, batch_size, logger
        )

        # Split training data -> Training/validation
        # Standard split for training and validation
        train_loader, val_loader = dutils.split_data(
            train_data, data_layer, batch_size=batch_size, logger=logger
        )

        prev_upd = 0
        for epoch in range(num_epochs):
            if logger is not None:
                logger.info(f">>> EPOCH {epoch + 1}/{num_epochs}")

            prev_upd = train_epoch(
                fold, prev_upd, model, train_loader, optimizer, device, writer, logger
            )
            val_epoch(fold, prev_upd, model, val_loader, device, writer, logger)

        # Test on fold
        # If model performs good on completly unseen data it generalizes good
        # Guarantees that no donor specific batch effects are learned
        test_fold(fold, model, test_loader, device, writer, logger)


def train_epoch(
    fold: str,
    prev_upd: int,
    model: Autoencoder,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
) -> int:
    """Trains the model for one epoch.

    Args:
        fold: The current fold identifier.
        prev_upd: The number of updates from previous epochs.
        model: The autoencoder model being trained.
        train_loader: DataLoader for the training data.
        optimizer: The optimizer being used for training.
        device: The device to run the training on ("cpu" or "cuda"). Defaults to "cpu".
        writer: An optional SummaryWriter object for TensorBoard logging.
        logger: An optional Logger object for logging output.

    Returns:
        int: The total number of updates after this epoch.
    """
    model.train()

    for batch_idx, (batch, _) in enumerate(tqdm(train_loader)):
        num_upd = prev_upd + batch_idx

        batch = batch.to(device)

        outputs = model(batch)
        loss = outputs.loss

        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation

        if num_upd % 100 == 0:
            # Calculate total gradinet norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            if writer is not None:
                if not hasattr(outputs, "loss_recon") or not hasattr(
                    outputs, "loss_kl"
                ):
                    # Log AE metrics
                    log_train_metrics(writer, fold, total_norm, loss.item(), num_upd)
                else:
                    # Log VAE metrics
                    log_train_metrics(
                        writer,
                        fold,
                        total_norm,
                        loss.item(),
                        num_upd,
                        outputs.loss_recon.item(),
                        outputs.loss_kl.item(),
                    )

            if logger is not None:
                logger.info(
                    f"Step {num_upd:,} (N samples: {num_upd * train_loader.batch_size:,}), Loss: {loss.item():.4f}, Grad: {total_norm:.4f}"
                )

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the model parameters
        # Needs to be after gradient norm calculation
        optimizer.step()

    return prev_upd + len(train_loader)


def val_epoch(
    fold: str,
    prev_upd: int,
    model: Autoencoder,
    val_loader: DataLoader,
    device: str = "cpu",
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
) -> None:
    """Validates the model for one epoch.

    Args:
        fold: The current fold identifier.
        prev_upd: The number of updates from previous epochs.
        model: The autoencoder model being validated.
        val_loader: DataLoader for the validation data.
        device: The device to run the validation on ("cpu" or "cuda"). Defaults to "cpu".
        writer: An optional SummaryWriter object for TensorBoard logging.
        logger: An optional Logger object for logging output.

    Returns:
        None
    """
    model.eval()

    val_loss, val_recon_loss, val_kl_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for data, _ in tqdm(val_loader, desc="Validation"):
            data = data.to(device)

            outputs = model(data)

            val_loss += outputs.loss.item()
            if hasattr(outputs, "loss_recon") and hasattr(outputs, "loss_kl"):
                val_recon_loss += outputs.loss_recon.item()
                val_kl_loss += outputs.loss_kl.item()

    # Avg validation loss
    val_loss /= len(val_loader)
    if val_recon_loss != 0.0 and val_kl_loss != 0.0:
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)

    if writer is not None:
        if val_recon_loss == 0.0 and val_kl_loss == 0.0:
            # Log AE metrics
            log_val_metrics(writer, fold, val_loss, prev_upd)
        else:
            # Log VAE metrics
            log_val_metrics(
                writer, fold, val_loss, prev_upd, val_recon_loss, val_kl_loss
            )

    if logger is not None:
        logger.info(f">>> VALIDATION Loss: {val_loss:.4f}")


def test_fold(
    fold: str,
    model: Autoencoder,
    test_loader: DataLoader,
    device: str = "cpu",
    writer: Optional[SummaryWriter] = None,
    logger: Optional[Logger] = None,
) -> None:
    """
    Tests the model on a specific fold.

    Args:
        fold: The current fold identifier.
        model: The autoencoder model being tested.
        test_loader: DataLoader for the test data.
        device: The device to run the testing on ("cpu" or "cuda"). Defaults to "cpu".
        writer: An optional SummaryWriter object for TensorBoard logging.
        logger: An optional Logger object for logging output.

    Returns:
        None
    """
    model.eval()

    recons, latent_reps = [], []
    test_loss, test_recon_loss, test_kl_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)

            outputs = model(batch)

            test_loss += outputs.loss.item()
            if hasattr(outputs, "loss_recon") and hasattr(outputs, "loss_kl"):
                test_recon_loss += outputs.loss_recon.item()
                test_kl_loss += outputs.loss_kl.item()

            # Data for performance plotting
            recons.extend(list(zip(batch, outputs.x_recon, labels)))
            latent_reps.extend(list(zip(outputs.z_sample, labels)))

    # Avg test loss
    test_loss /= len(test_loader)
    if test_recon_loss != 0.0 and test_kl_loss != 0.0:
        test_recon_loss /= len(test_loader)
        test_kl_loss /= len(test_loader)

    if writer is not None:
        if test_recon_loss == 0.0 and test_kl_loss == 0.0:
            # Log AE metrics
            log_val_metrics(writer, fold, test_loss)
        else:
            # Log VAE metrics
            log_val_metrics(writer, fold, test_loss, test_recon_loss, test_kl_loss)
        dutils.plot_recon_performance(recons, scope="Sample", method="Sum", fold=fold)
        dutils.plot_recon_performance(recons, scope="Sample", method="Mean", fold=fold)
        dutils.plot_recon_performance(recons, scope="Gene", method="Sum", fold=fold)
        dutils.plot_recon_performance(recons, scope="Gene", method="Mean", fold=fold)
        dutils.plot_latent_space(latent_reps)

    if logger is not None:
        logger.info(f">>> TEST Loss: {test_loss:.4f}")
