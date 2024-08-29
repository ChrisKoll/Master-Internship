"""
Sparse Data Processing Module.

This module provides utilities for handling sparse single-cell data in a PyTorch
environment. It includes a custom PyTorch dataset class for sparse matrices and functions
to split data for training and testing, including support for k-fold cross-validation.

Classes:
    - SparseDataset: A custom PyTorch Dataset class for sparse data.

Functions:
    - split_data: Splits a tensor into training and validation datasets.
    - create_outer_fold: Creates training and test folds based on a donor column.
    - plot_recon_performance: Generates a scatter plot comparing original and reconstructed data.
    - plot_latent_space: Generates a scatter plot of the PCA-transformed latent space.
"""

# Standard imports
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


class SparseDataset(Dataset):
    """
    Custom dataset class for sparse data.

    This class provides an interface for working with sparse data in PyTorch.
    It supports indexing and retrieval of rows from sparse matrices, converting
    them into dense PyTorch tensors.

    Attributes:
        sparse_data (csr_matrix): A SciPy sparse matrix (CSR format) containing the data.
        anno_cell_type (List[str]): List of annotation labels corresponding to the data rows.

    Args:
        sparse_data (csr_matrix): A sparse matrix in Compressed Sparse Row (CSR) format.
        anno_cell_type (List[str]): Annotation labels corresponding to the data rows.
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


def split_data(
    adata: AnnData,
    data_layer: str,
    train_dist: float = 0.8,
    batch_size: int = 128,
    logger: Optional[Logger] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Splits the data from an AnnData object into training and validation datasets.

    This function randomly splits the data into training and validation sets based on the
    specified distribution ratio and creates PyTorch DataLoaders for each set. The training
    and validation sets are used to train and evaluate machine learning models, respectively.

    Args:
        adata: An AnnData object containing the data and metadata. The data to be split
            is accessed via `adata.layers[data_layer]`, and the corresponding labels are
            accessed via `adata.obs["cell_type"]`.
        data_layer: The key in `adata.layers` where the input data is stored. Each row
            represents a sample, and each column represents a feature.
        train_dist: The proportion of the data to allocate to the training set. Defaults
            to 0.8, meaning 80% of the data will be used for training and 20% for validation.
        batch_size: The batch size for the DataLoaders. Defaults to 128.
        logger: An optional Logger instance for logging details about the data split. If
            provided, it logs information about the split and the number of entries in each set.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing two DataLoaders. The first DataLoader
        is for the training set, and the second is for the validation set.
    """
    # Calculate the number of entries (from percentage), used as training data
    # Default value -> 80%
    train_size = int(train_dist * adata.shape[0])

    # Set seed to make data distribution reproducable
    # torch.manual_seed(19082024)
    perm = torch.randperm(adata.shape[0])
    train_split, val_split = perm[:train_size], perm[train_size:]

    if logger is not None:
        logger.debug("Distributed entries into training/validation splits")

    # Create sparse datasets from the split data
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
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


@DeprecationWarning
def split_data_kfcv(
    adata: AnnData,
    layer_name: str,
    train_dist: float = 0.8,
    batch_size: int = 128,
    logger: Optional[Logger] = None,
) -> tuple[AnnData, DataLoader]:
    """
    Currently deprecated.
    """
    # Calculate the number of entries (from percentage), used as training data
    # Default value -> 80%
    train_size = int(train_dist * adata.shape[0])

    # Set seed to make data distribution reproducable
    torch.manual_seed(19082024)
    perm = torch.randperm(adata.shape[0])
    train_split, test_split = perm[:train_size], perm[train_size:]

    if logger is not None:
        logger.info("Distributed entries into training/test splits")

    # Keep adata object for k-fold manipulation
    # Only convert test data to sparse dataset
    train_data = adata[train_split.tolist(), :]
    test_data = SparseDataset(
        adata.layers[layer_name][test_split, :],
        adata.obs["cell_type"].iloc[test_split.tolist()].tolist(),
    )

    if logger is not None:
        logger.info(f"Training split created. Contains {len(train_split)} entries")
        logger.info(f"Test split created. Contains {len(test_split)} entries")

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
    )

    return train_data, test_loader


def create_outer_fold(
    adata: AnnData,
    donor: str,
    data_layer: str,
    batch_size: int = 128,
    logger: Optional[Logger] = None,
) -> tuple[AnnData, DataLoader]:
    """
    Creates training and validation folds based on a specified donor.

    This function splits the data into training and validation sets according to a specified
    donor identifier in the AnnData object. It generates DataLoaders for the training and
    validation sets, allowing for efficient model training and evaluation.

    Args:
        adata: An AnnData object containing the dataset. The data is accessed via `adata.layer[data_layer]`, and the donor information is found in `adata.obs["donor"]`.
        donor: The donor identifier to use for selecting the validation set. Data from this donor is used as the validation set, while data from other donors is used for training.
        data_layer: The key in `adata.layers` that contains the data to be used for training and validation.
        batch_size: The batch size for the DataLoaders. Defaults to 128.
        logger: An optional Logger instance for logging details about the data split. If provided, it logs information about the number of entries in each fold.

    Returns:
        tuple[AnnData, DataLoader]: A tuple containing two DataLoaders. The first DataLoader is for the training set, and the second is for the validation set (test fold).
    """
    # Remains adata for further splitting
    # Cell type information is needed
    train_data = adata[adata.obs["donor"] != donor]

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
        shuffle=False,
    )

    return train_data, test_loader


def plot_recon_performance(
    plotting_data: list[tuple[torch.Tensor, torch.Tensor, str]],
    scope: Literal["Sample", "Gene"] = "Sample",
    method: Literal["Sum", "Mean"] = "Sum",
    fold: str = "",
):
    """
    Generates a scatter plot comparing original and reconstructed data based on the specified scope and method and saves it to file.

    Args:
        plotting_data (list[tuple[torch.Tensor, torch.Tensor, str]]): List of tuples where each tuple contains the original data tensor, the reconstructed data tensor, and the label.
        scope (Literal["Sample", "Gene"], optional): Scope of the plot, either "Sample" or "Gene".Defaults to "Sample".
        method (Literal["Sum", "Mean"], optional): Method of summarization, either "Sum" or "Mean". Defaults to "Sum".
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

        if scope == "Sample":
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

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{fold}_performance_{scope.lower()}_{method.lower()}.png")


def plot_latent_space(plotting_data: list[tuple[torch.Tensor, str]], fold: str = ""):
    """
    Generates a scatter plot of the PCA-transformed latent space and saves it to file.

    Args:
        plotting_data (list[tuple[torch.Tensor, str]]): List of tuples where each tuple contains
            the latent space tensor and the label.
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

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{fold}_latent_space.png")
