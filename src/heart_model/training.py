# Standard library imports
from typing import Optional

# Third-party library imports
from anndata import AnnData

# Local/application-specific import
from src import model


def train(data: AnnData, vae: Optional[model.VariationalAutoencoder] = None, donors: Optional[list[str]] = None,
          *, epochs: int, device: str):
    """
    :param vae: Model architecture to be trained
    :param data: Data the model will be trained on
    :param donors: List of all donors
    :param epochs: Number of epochs
    :param device: Provides the device for training
    :return: Returns the trained VAE model
    """
    # Shuffle the dataset ?