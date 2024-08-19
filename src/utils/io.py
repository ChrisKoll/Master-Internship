"""Docstring."""

# Standard imports
from logging import Logger
from os.path import basename
from typing import Optional

# Third-party imports
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix
from torch import Tensor

__author__ = "Christian Kolland"
__version__ = "0.1"


def load_adata(path_to_data: str, logger: Optional[Logger] = None) -> AnnData:
    """Docstring."""
    if logger is not None:
        logger.info(f"Start loading dataset: {basename(path_to_data)}")

    adata = read_h5ad(filename=path_to_data)

    if logger is not None:
        logger.info(f"Dataset loaded")

    return adata


def load_adata_layer(path_to_data: str, layer: str, logger: Logger) -> Tensor:
    """Docstring."""
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
    """Docstring."""
    adata.layers[layer_name] = transformed_data
    adata.write_h5ad(filename=filename)

    if logger is not None:
        logger.info(f"Added layer of transformed data: {layer_name}")
