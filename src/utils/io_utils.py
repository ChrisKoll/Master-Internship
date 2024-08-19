"""
Data I/O Utilities Module.

This module provides utility functions for loading and saving single-cell data using the AnnData format. 
It includes functions for loading entire datasets, loading specific layers of data, and saving transformed data layers.

Functions:
    - load_adata: Loads an AnnData object from a specified file path.
    - load_adata_layer: Loads a specific layer of an AnnData object as a PyTorch tensor.
    - save_adata_layer: Saves a transformed data layer into an AnnData object and writes it to a file.
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
    Load an AnnData object from a specified file path.

    This function reads an AnnData (.h5ad) file and returns the corresponding AnnData object.
    Optionally logs the loading process using the provided logger.

    Args:
        path_to_data (str): Path to the .h5ad file containing the single-cell data.
        logger (Optional[Logger]): Logger instance for logging information. Defaults to None.

    Returns:
        AnnData: The loaded AnnData object.

    Example:
        >>> adata = load_adata("data/input_data.h5ad", logger)
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
    Load a specific layer of an AnnData object as a PyTorch tensor.

    This function reads an AnnData (.h5ad) file, extracts a specified data layer,
    and returns it as a PyTorch tensor. Optionally logs the process using the provided logger.

    Args:
        path_to_data (str): Path to the .h5ad file containing the single-cell data.
        layer (str): Name of the layer to be extracted from the AnnData object.
        logger (Optional[Logger]): Logger instance for logging information. Defaults to None.

    Returns:
        Tensor: The specified layer of the AnnData object as a PyTorch tensor.

    Example:
        >>> layer_data = load_adata_layer("data/input_data.h5ad", "transformed_data", logger)
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
    Save a transformed data layer into an AnnData object and write it to a file.

    This function adds a new layer containing transformed data to an AnnData object and writes the
    updated object to a specified .h5ad file. Optionally logs the saving process using the provided logger.

    Args:
        adata (AnnData): The AnnData object to which the new layer will be added.
        transformed_data (csr_matrix): The transformed data to be added as a new layer.
        layer_name (str): The name of the new layer to be added to the AnnData object.
        filename (str): The file path where the updated AnnData object will be saved.
        logger (Optional[Logger]): Logger instance for logging information. Defaults to None.

    Returns:
        None

    Example:
        >>> save_adata_layer(adata, transformed_data, "layer_name", "output_data.h5ad", logger)
    """
    adata.layers[layer_name] = transformed_data
    adata.write_h5ad(filename=filename)

    if logger is not None:
        logger.info(f"Added layer of transformed data: {layer_name}")
