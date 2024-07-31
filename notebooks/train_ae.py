"""
Main module that runs all the code.

This script loads the data, configuration, splits the data, initializes the model, and trains the model.
"""

# Standard imports
import argparse
from datetime import datetime
import os

# Third-party imports
import anndata as ad
import torch
from torch.utils.tensorboard import SummaryWriter

# Self-built modules
import modules.autoencoder as ae
from modules.logging_setup import main_logger
from modules.sparse_dataset import SparseDataset
import modules.ae_training as T
import modules.utils as utils

__author__ = "Christian Kolland"
__version__ = "0.0.1"


def load_data(data_path: str) -> torch.Tensor:
    """
    Loads the data from the given file path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        torch.Tensor: Normalized data.
    """
    # Load AnnData object
    adata = ad.read_h5ad(filename=data_path)

    # Logging
    main_logger.info(f"Loaded dataset: {os.path.basename(data_path)}")

    # Return normalized data
    return adata.layers["min_max_normalized"]


def load_config(config_path: str) -> dict:
    """
    Loads the model configuration from the given file path.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Model configuration.
    """
    model_config = utils.import_model_params(config_path)

    # Logging
    main_logger.info(f"Loaded model config: {os.path.basename(config_path)}")

    return model_config


def split_data(count_data: torch.Tensor, batch_size: int) -> tuple:
    """
    Splits the data into training and testing sets.

    Args:
        count_data (torch.Tensor): Count matrix.
        batch_size (int): Batch size for the data loaders.

    Returns:
        tuple: Training and testing data loaders.
    """
    train_size = int(0.8 * count_data.shape[0])
    ## TODO: Remove unused data
    # test_size = count_data.shape[0] - train_size

    torch.manual_seed(2406)
    perm = torch.randperm(count_data.shape[0])
    train_split, test_split = perm[:train_size], perm[train_size:]

    # Logging
    main_logger.info("Split data to training/testing")

    train_data = SparseDataset(count_data[train_split, :])
    test_data = SparseDataset(count_data[test_split, :])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def load_model(model_architecture: dict) -> ae.Autoencoder:
    """
    Initializes the model based on the given architecture.

    Args:
        model_architecture (dict): Model architecture configuration.

    Returns:
        ae.Autoencoder: Initialized autoencoder model.
    """
    # Assemble model from config
    encoder_layers, decoder_layers = utils.import_model_architecture(
        forward=model_architecture["layers"]["encoder"],
        backward=model_architecture["layers"]["decoder"],
    )

    # Import loss function
    loss_function = utils.import_loss_function(model_architecture["loss_function"])

    ## Create model
    model = ae.Autoencoder(encoder_layers, decoder_layers, loss_function=loss_function)

    # Logging
    main_logger.info("Initialized model")

    return model


def train_model(
    model: ae.Autoencoder,
    optimization: dict,
    version: str,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> None:
    """
    Trains the model.

    Args:
        model (ae.Autoencoder): The autoencoder model to train.
        optimization (dict): Optimization configuration.
        version (str): Model version.
        num_epochs (int): Number of training epochs.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        test_loader (torch.utils.data.DataLoader): Testing data loader.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    """
    # Import optimizer
    optimizer = utils.import_optimizer(
        model.parameters(),
        optimization["optimizer"],
        learning_rate=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
    )

    writer = SummaryWriter(
        f'../runs/hca/ae_{version}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    prev_updates = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        prev_updates = T.train(
            model, train_loader, optimizer, prev_updates, device, writer=writer
        )
        T.test(model, test_loader, prev_updates, device=device, writer=writer)


def main() -> None:
    """
    Main function to run the entire process: data loading, configuration loading, data splitting, model initialization, and training.
    """
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the data file.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    # Established the type of device used for model processing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data = load_data(args.data)

    model_config = load_config(args.config)
    # Split into model and training
    model_architecture = model_config["model"]
    model_training = model_config["training"]

    # Split data
    train_loader, test_loader = split_data(data, model_training["batch_size"])

    # Assemble model
    model = load_model(model_architecture)

    # Train model
    train_model(
        model,
        model_architecture["optimization"],
        model_architecture["version"],
        model_training["training_epochs"],
        train_loader,
        test_loader,
        device,
    )


if __name__ == "__main__":
    main()
