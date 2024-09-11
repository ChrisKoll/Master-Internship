"""
Utility Functions for Sparse Data Processing.

This module provides tools for processing sparse single-cell data in a PyTorch
environment, specifically designed for Autoencoder training workflows. It includes
a custom PyTorch Dataset class for handling sparse matrices and functions for
data splitting and evaluation, including support for k-fold cross-validation.

Classes:
    - TrainingConfig: Stores configuration parameters for model training, including 
        batch size, number of epochs, and device information.
    - SparseDataset: A custom PyTorch Dataset class for working with sparse data in 
        compressed sparse row (CSR) format.

Functions:
    - split_data: Splits a dataset into training and validation sets, returning 
        PyTorch DataLoaders for each set.
    - create_outer_fold: Generates training and test folds based on donor IDs.
    - plot_recon_performance: Generates scatter plots comparing original data to 
        reconstructed data using either 'Sum' or 'Mean' methods.
    - plot_latent_space: Visualizes the latent space of an Autoencoder using PCA 
      and generates a scatter plot.
"""

# Standard imports
from dataclasses import dataclass
from logging import Logger
import os
from typing import Literal, Optional

# Third-party imports
from anndata import AnnData
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

__author__ = "Christian Kolland"
__version__ = "1.0"

# Constants
_DIM_RED = {
    "Sample": {"Sum": lambda x: x.sum(axis=1), "Mean": lambda x: x.mean(axis=1)},
    "Gene": {"Sum": lambda x: x.sum(axis=0), "Mean": lambda x: x.mean(axis=0)},
}

# Non GUI-backend
matplotlib.use("Agg")


@dataclass
class TrainingConfig:
    """
    Configuration class for storing parameters related to model training.

    Attributes:
        folds (list[str]): List of folds or donor identifiers for data splitting.
        batch_size (int): The number of samples processed in each training step.
        num_epochs (int): The total number of iterations for training.
        device (str): The computational device to use, either 'cuda' for GPU
                      or 'cpu'.
    """

    folds: list[str]
    batch_size: int
    num_epochs: int
    device: str


class SparseDataset(Dataset):
    """
    A custom PyTorch Dataset class for working with sparse data in CSR format.

    This class allows sparse single-cell data to be efficiently accessed and
    processed in PyTorch. It provides functionality for retrieving individual
    rows of sparse matrices as dense PyTorch tensors along with their associated
    labels.
    """

    def __init__(self, sparse_data: csr_matrix, anno_cell_type: list[str]):
        """
        Initializes the SparseDataset with sparse data.

        Args:
            sparse_data (csr_matrix): The sparse matrix data to be used by the dataset.
            anno_cell_type (List[str]): Annotation labels corresponding to the data rows.
        """
        self.sparse_data = sparse_data
        self.anno_cell_type = anno_cell_type

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of rows in the sparse matrix.
        """
        return self.sparse_data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, str]: A tuple containing the data for the specified index and its label.

        Raises:
            IndexError: If the index is out of range.
        """
        if index >= len(self):
            raise IndexError("Index out of range")

        # Extract the row as a dense numpy array
        row = self.sparse_data.getrow(index).toarray().squeeze()

        # Retrieve the corresponding annotation label
        label = self.anno_cell_type[index]

        # Convert the row to a PyTorch tensor
        return torch.tensor(row, dtype=torch.float32), label


def create_outer_fold(
    adata: AnnData,
    donor: str,
    data_layer: str,
    batch_size: int = 128,
    logger: Optional[Logger] = None,
) -> tuple[AnnData, DataLoader]:
    """
    Splits data into training and validation folds based on a specific donor.

    This function isolates data from a specified donor for validation, and uses
    data from other donors for training. It returns the training data and a
    DataLoader for the validation fold.

    Args:
        adata (AnnData): The dataset to split, containing data and donor information.
        donor (str): The donor identifier for the validation set.
        data_layer (str): The data layer to use for training and validation.
        batch_size (int, optional): The number of samples per batch. Defaults to 128.
        logger (Optional[Logger], optional): Logger instance for progress tracking.

    Returns:
        tuple[AnnData, DataLoader]: Training data and validation DataLoader.
    """
    # Training split containing all donors but one
    # Remains adata for further splitting (train/val)
    train_data = adata[adata.obs["donor"] != donor]

    # Test split contains one donor (small sample size)
    # If model performs well on this data -> Good training
    test_split = adata[adata.obs["donor"] == donor]
    test_data = SparseDataset(
        test_split.layers[data_layer],
        test_split.obs["cell_type"].tolist(),
    )

    if logger is not None:
        logger.info(f"Training fold created. Contains {train_data.shape[0]} entries")
        logger.info(f"Test fold created. Contains {test_split.shape[0]} entries")

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # No shuffeling for testing
        num_workers=8,  # Improves processing
    )

    return train_data, test_loader


def split_data(
    adata: AnnData,
    data_layer: str,
    train_dist: float = 0.8,
    batch_size: int = 128,
    logger: Optional[Logger] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Splits an AnnData object into training and validation sets and returns PyTorch DataLoaders.

    The function randomly splits the dataset into training and validation sets based
    on the specified distribution ratio. DataLoaders are created for both sets to
    facilitate batched training and evaluation.

    Args:
        adata (AnnData): An AnnData object containing single-cell data.
        data_layer (str): The layer of the AnnData object that contains the data to be split.
        train_dist (float, optional): The proportion of the data used for training. Defaults to 0.8.
        batch_size (int, optional): The number of samples per batch. Defaults to 128.
        logger (Optional[Logger], optional): Logger instance for tracking progress.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation DataLoaders for PyTorch models.
    """
    # Calculate the number of entries (from percentage), used as training data
    # Default value -> 80%
    train_size = int(train_dist * adata.shape[0])

    # Set seed to make data distribution reproducable
    torch.manual_seed(19082024)
    perm = torch.randperm(adata.shape[0])
    train_split, val_split = perm[:train_size], perm[train_size:]

    if logger is not None:
        logger.debug("Distributed entries into training/validation splits")

    train_data = SparseDataset(
        adata.layers[data_layer][train_split, :],
        adata.obs["cell_type"].iloc[train_split.tolist()].tolist(),
    )
    val_data = SparseDataset(
        adata.layers[data_layer][val_split, :],
        adata.obs["cell_type"].iloc[val_split.tolist()].tolist(),
    )

    if logger is not None:
        logger.info(f"Training split created. Contains {len(train_split)} entries")
        logger.info(f"Validation split created. Contains {len(val_split)} entries")

    # Shuffle training data to prevent batch effect
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to improve learning
        num_workers=8,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return train_loader, val_loader


def plot_recon_performance(
    plotting_data: list[tuple[torch.Tensor, torch.Tensor, str]],
    dir: str,
    scope: Literal["Sample", "Gene"] = "Sample",
    method: Literal["Sum", "Mean"] = "Sum",
    fold: str = "",
    logger: Optional[Logger] = None,
) -> None:
    """
    Plots the reconstruction performance of the Autoencoder by comparing
    original and reconstructed data.

    This function generates scatter plots comparing the original and reconstructed
    data using either a 'Sum' or 'Mean' operation. It supports both sample-level and
    gene-level analyses and saves the plot as an image file.

    Args:
        plotting_data (list[tuple[torch.Tensor, torch.Tensor, str]]): A list of tuples
            containing the original data, reconstructed data, and corresponding labels.
        dir (str): Directory to save the generated plot.
        scope (Literal["Sample", "Gene"], optional): Whether to summarize by sample or gene.
        method (Literal["Sum", "Mean"], optional): Summarization method to apply. Defaults to "Sum".
        fold (str, optional): Fold identifier for file naming. Defaults to an empty string.
        logger (Optional[Logger], optional): Logger instance for logging correlation data.

    Returns:
        None: Saves the generated plot to file.
    """
    # Unpack plotting data and convert tensors to numpy arrays
    xs, x_hats, labels = zip(*plotting_data)
    xs = np.stack([x.detach().numpy() for x in xs])
    x_hats = np.stack([x_hat.detach().numpy() for x_hat in x_hats])

    # Create a DataFrame for storing the results
    recon_df = pd.DataFrame()

    # Check if the scope and method are valid and apply the selected operation
    if scope in _DIM_RED and method in _DIM_RED[scope]:
        summarise = _DIM_RED[scope][method]
        recon_df["X"] = summarise(xs)
        recon_df["X_Hat"] = summarise(x_hats)

        if scope == "Gene":
            corrleation, p_value = pearsonr(recon_df["X"], recon_df["X_Hat"])

            if logger is not None:
                logger.info(f">>> Correlation: {corrleation}, p-Value: {p_value}")
        elif scope == "Sample":
            recon_df["Cell_Type"] = labels

    # Set Seaborn theme and palette
    sns.set_theme()
    sns.set_palette("Paired")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot based on scope
    if scope == "Sample":
        sns.scatterplot(x="X", y="X_Hat", data=recon_df, hue="Cell_Type", ax=ax)
        # Move legend out of plot
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    else:  # scope == "Gene"
        sns.scatterplot(x="X", y="X_Hat", data=recon_df, ax=ax)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Reconstructed")
    ax.set_title(f"Reconstruction Performance - {scope} {method}")

    # Makes legend not overflow
    fig.tight_layout()

    os.makedirs(f"plots/{dir}/", exist_ok=True)
    fig.savefig(f"plots/{dir}/{fold}_performance_{scope.lower()}_{method.lower()}.png")


def plot_latent_space(
    plotting_data: list[tuple[torch.Tensor, str]],
    dir: str,
    fold: str,
) -> None:
    """
    Visualizes the latent space learned by the Autoencoder using PCA.

    This function generates a 2D scatter plot of the PCA-transformed latent space,
    color-coded by cell type. It saves the plot as an image file.

    Args:
        plotting_data (list[tuple[torch.Tensor, str]]): List of latent space representations and labels.
        dir (str): Directory to save the plot.
        fold (str): Fold identifier for file naming.

    Returns:
        None: Saves the PCA plot to file.
    """
    zs, labels = zip(*plotting_data)
    # Transormation of zs to be processed by PCA
    zs = np.stack([z.detach().numpy() for z in zs])

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(zs)
    # Save variance explanation for each PC
    variance = pca.explained_variance_ratio_ * 100

    # Data frame for plotting
    pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2"])
    pca_df["Cell_Type"] = labels

    sns.set_theme()
    sns.set_palette("Paired")  # Color palette

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x="PC1", y="PC2", data=pca_df, hue="Cell_Type", ax=ax)
    # Move legend out of plot
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_xlabel(f"PC1 - {variance[0]:.1f}% Variance")
    ax.set_ylabel(f"PC2 - {variance[1]:.1f}% Variance")
    ax.set_title("PCA of AE Latent Space")

    fig.tight_layout()

    os.makedirs(f"plots/{dir}/", exist_ok=True)
    fig.savefig(f"plots/{dir}/{fold}_latent_space.png")
