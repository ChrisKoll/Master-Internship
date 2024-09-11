"""
Utility Functions for Data I/O.

This module provides utility functions for loading and saving single-cell data in the AnnData format. It supports loading entire datasets, extracting specific layers as PyTorch tensors, and saving transformed data back into AnnData objects.

Functions:
    - load_adata: Loads an AnnData object from a specified .h5ad file path.
    - load_adata_layer: Loads a specific data layer from an AnnData object as a PyTorch tensor.
    - save_adata_layer: Saves a transformed data layer into an AnnData object and writes it to 
        a h5ad file.
"""

# Standard imports
from logging import Logger
from os.path import basename
from typing import Optional

# Third-party imports
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix
from torch import Tensor

__author__ = "Christian Kolland"
__version__ = "1.0"


def load_adata(path_to_data: str, logger: Optional[Logger] = None) -> AnnData:
    """
    Loads an AnnData object from a specified file path.

    This function reads a h5ad file containing single-cell data and returns the loaded
    AnnData object. Optionally, a logger can be provided to log the loading process.

    Args:
        path_to_data (str): The file path to the .h5ad file.
        logger (Optional[Logger]): A logger instance for logging information about the loading
            process. If None, no logging will occur. Defaults to None.

    Returns:
        AnnData: The loaded AnnData object containing single-cell data and metadata.
    """
    if logger is not None:
        logger.info(f"Start loading dataset: {basename(path_to_data)}")

    adata = read_h5ad(filename=path_to_data)

    if logger is not None:
        logger.info(f"Dataset loaded")

    return adata


def load_adata_layer(
    path_to_data: str, layer: str, logger: Optional[Logger] = None
) -> Tensor:
    """
    Loads a specific layer from an AnnData object as a PyTorch tensor.

    This function reads a h5ad file, extracts a specified data layer from the AnnData object,
    and returns it as a PyTorch tensor. It is particularly useful for accessing transformed
    data layers stored in the AnnData object. Optionally, a logger can be provided to log the process.

    Args:
        path_to_data (str): The file path to the .h5ad file containing the AnnData object.
        layer (str): The name of the data layer to be extracted from the AnnData object.
        logger (Optional[Logger]): A logger instance for logging information about the loading
            process. If None, no logging will occur. Defaults to None.

    Returns:
        Tensor: A PyTorch tensor containing the data from the specified layer.
    """
    if logger is not None:
        logger.info(f"Start loading dataset: {basename(path_to_data)}")

    adata = read_h5ad(filename=path_to_data)

    if logger is not None:
        logger.info(f"Dataset loaded. Layer: {layer}")

    # Return one layer (in most cases: transformed data)
    # No other counts or metadata needed
    return adata.layers[layer]


def save_adata_layer(
    adata: AnnData,
    transformed_data: csr_matrix,
    layer_name: str,
    filename: str,
    logger: Optional[Logger] = None,
) -> None:
    """
    Saves a transformed data layer into an AnnData object and writes it to a specified h5ad file.

    This function adds a new data layer to the provided AnnData object and writes the updated
    AnnData object to a file in h5ad format. The new layer can contain any transformed data,
    such as denoised or normalized counts. Optionally, a logger can be provided to log the saving process.

    Args:
        adata (AnnData): The AnnData object to which the new layer will be added.
        transformed_data (csr_matrix): The transformed data to be stored as a new layer.
        layer_name (str): The name of the new data layer.
        filename (str): The file path where the updated AnnData object will be saved.
        logger (Optional[Logger]): A logger instance for logging information about the saving
            process. If None, no logging will occur. Defaults to None.
    """
    adata.layers[layer_name] = transformed_data
    adata.write_h5ad(filename=filename)

    if logger is not None:
        logger.info(f"Added layer of transformed data: {layer_name}")
