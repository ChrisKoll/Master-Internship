"""Docstring."""

# Standard imports
from datetime import datetime

# Third-party imports
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Self-built modules
from utils.data_utils import create_fold, plot_expression_profiles
import utils.json_utils as jutils

__author__ = "Christian Kolland"
__version__ = 0.1


def fit(
    version,
    model,
    train_data,
    data_layer,
    test_loader,
    optimizer,
    batch_size=128,
    num_epochs=50,
    device="cpu",
    logger=None,
):
    """Docstring."""
    optimizer = jutils.configure_optimizer(
        model.parameters(),
        optimizer.optimizer,
        learning_rate=optimizer.learning_rate,
        weight_decay=optimizer.weight_decay,
    )

    writer = SummaryWriter(
        f'runs/hca/ae_{version}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    donors = ["D1", "H2", "D5"]
    train_fold(
        model,
        donors,
        train_data,
        data_layer,
        test_loader,
        optimizer,
        batch_size,
        num_epochs,
        device,
        writer,
        logger,
    )

    writer.close()


def train_fold(
    model,
    folds,
    train_data,
    data_layer,
    test_loader,
    optimizer,
    batch_size=128,
    num_epochs=50,
    device="cpu",
    writer=None,
    logger=None,
):
    """Docstring."""
    for fold_idx, fold in enumerate(folds):
        if logger is not None:
            logger.info(f">>> FOLD {fold_idx + 1}/{len(folds)} - {fold}")

        train_loader, val_loader = create_fold(
            train_data, fold, data_layer, batch_size, logger
        )

        prev_upd = 0
        for epoch in range(num_epochs):
            if logger is not None:
                logger.info(f">>> EPOCH {epoch + 1}/{num_epochs}")

            prev_upd = train_epoch(
                fold, prev_upd, model, train_loader, optimizer, device, writer, logger
            )
            val_epoch(fold, prev_upd, model, val_loader, device, writer, logger)

        test_fold(fold, model, test_loader, device, writer, logger)


def train_epoch(
    fold,
    prev_upd,
    model,
    train_loader,
    optimizer,
    device="cpu",
    writer=None,
    logger=None,
):
    """Docstring."""
    model.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
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

            logger.info(
                f"Step {num_upd:,} (N samples: {num_upd * train_loader.batch_size:,}), Loss: {loss.item():.4f}, Grad: {total_norm:.4f}"
            )

            if writer is not None:
                writer.add_scalar(f"{fold}/Train/GradNorm", total_norm, num_upd)
                writer.add_scalar(f"{fold}/Train/Loss", loss.item(), num_upd)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the model parameters
        # Needs to be after gradient norm calculation
        optimizer.step()

    return prev_upd + len(train_loader)


def val_epoch(
    fold, prev_upd, model, val_loader, device="cpu", writer=None, logger=None
):
    "Docstring."
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.to(device)

            outputs = model(data)
            loss = outputs.loss
            val_loss += loss.item()

    val_loss /= len(val_loader)

    if writer is not None:
        writer.add_scalar(f"{fold}/Val/Loss", val_loss, prev_upd)

    if logger is not None:
        logger.info(f">>> VALIDATION Loss: {val_loss:.4f}")


def test_fold(fold, model, test_loader, device="cpu", writer=None, logger=None):
    """Docstring."""
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            batch = batch.to(device)

            outputs = model(batch)
            loss = outputs.loss

            test_loss += loss.item()

            if batch_idx % 5 == 0:
                if writer is not None:
                    writer.add_figure(
                        f"{fold}/Test/Recon",
                        plot_expression_profiles(batch[0], outputs.x_recon[0]),
                    )

                    # TODO: Latent Space visualization

    test_loss /= len(test_loader)

    if writer is not None:
        writer.add_scalar(f"{fold}/Test/Loss", test_loss)

    logger.info(f">>> TEST Loss: {test_loss:.4f}")
