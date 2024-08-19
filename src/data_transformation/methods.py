"""
Normalization and Transformation of Gene Expression Count Matrices.

This module provides functions to normalize and transform gene expression count matrices
in both dense (numpy array) and sparse (scipy sparse matrix) formats. The functions available
include Counts Per Million (CPM) normalization, natural logarithm transformation, and min-max scaling.

Functions:
    - dense_cpm: Normalize a dense count matrix using Counts Per Million (CPM).
    - dense_log: Apply a natural logarithm transformation to a dense count matrix.
    - dense_min_max: Rescale a dense count matrix to a specified range using min-max scaling.
    - sparse_cpm: Normalize a sparse count matrix using Counts Per Million (CPM).
    - sparse_log: Apply a natural logarithm transformation to a sparse count matrix.
    - sparse_min_max: Rescale a sparse count matrix to a specified range using min-max scaling.
"""

# Standard imports
from logging import Logger
from typing import Optional

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm

__author__ = "Christian Kolland"
__version__ = 1.0

# Constants
_CPM_SCALING_FACT = 1e6
_EPSILON = 1e-9  # Small constant to prevent division by zero


def dense_cpm(count_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a dense count matrix using Counts Per Million (CPM).

    This function calculates the CPM normalization for a dense count matrix where each row represents a sample/cell and each column represents a gene.

    Args:
        count_matrix (np.ndarray): Dense count matrix (rows = samples/cells, cols = genes).

    Returns:
        np.ndarray: CPM-normalized count matrix.

    Example:
        >>> count_matrix = np.array([[100, 200, 300], [400, 500, 600]])
        >>> dense_cpm(count_matrix)
        array([[100000., 200000., 300000.],
               [266666.66666667, 333333.33333333, 400000.]])
    """
    # Sums over all rows to get total counts
    cell_counts = np.sum(count_matrix, axis=1)

    # Calculate the CPM normalized values
    cpm_normalized = count_matrix * _CPM_SCALING_FACT / cell_counts[:, np.newaxis]

    return cpm_normalized


def dense_log(count_matrix: np.ndarray) -> np.ndarray:
    """
    Apply a natural logarithm transformation to a dense count matrix.

    This function calculates the natural log transformation for a dense count matrix, adding 1 to each value to handle zero counts.

    Args:
        count_matrix (np.ndarray): Dense count matrix (rows = samples/cells, cols = genes).

    Returns:
        np.ndarray: Log-transformed count matrix.

    Example:
        >>> count_matrix = np.array([[0, 2, 3], [4, 0, 6]])
        >>> dense_log(count_matrix)
        array([[0.        , 1.09861229, 1.38629436],
               [1.60943791, 0.        , 1.94591015]])
    """
    # Calculate log values
    # -> +1 to each value, to prevent expression = 0
    log_transformed = np.log(count_matrix + 1)

    return log_transformed


def dense_min_max(
    count_matrix: np.ndarray, min_val: int = 0, max_val: int = 1
) -> np.ndarray:
    """
    Rescale a dense count matrix to a specified range using min-max scaling.

    This function rescales the values for each feature (gene) in a dense count matrix to the specified range.

    Args:
        count_matrix (np.ndarray): Dense count matrix (rows = samples/cells, cols = genes).
        min_val (float, optional): Minimum value after rescaling. Defaults to 0.
        max_val (float, optional): Maximum value after rescaling. Defaults to 1.

    Returns:
        np.ndarray: Min-max rescaled count matrix.

    Example:
        >>> count_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        >>> dense_min_max(count_matrix, 0, 1)
        array([[0. , 0. , 0. ],
               [1. , 1. , 1. ]])
    """
    # Determines min value for each matrix column
    min_data = np.min(count_matrix, axis=0)
    # Determines max value for each matrix column
    max_data = np.max(count_matrix, axis=0)

    # Rescales values to given range
    min_max_scaled = (count_matrix - min_data) / (max_data - min_data) * (
        max_val - min_val
    ) + min_val

    return min_max_scaled


def sparse_cpm(sp_matrix: csr_matrix, logger: Optional[Logger] = None) -> csr_matrix:
    """
    Normalize a sparse count matrix using Counts Per Million (CPM).

    This function calculates the CPM normalization for a sparse count matrix (Compressed Sparse Row format),
    where each row represents a sample/cell and each column represents a gene.

    Args:
        sp_matrix (csr_matrix): Sparse count matrix (CSR format).
        logger (Optional[Logger]): Logger for progress messages. Defaults to None.

    Returns:
        csr_matrix: CPM-normalized sparse count matrix.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> count_matrix = csr_matrix([[100, 200, 300], [400, 500, 600]])
        >>> sparse_cpm(count_matrix)
        <2x3 sparse matrix of type '<class 'numpy.float64'>'
            with 6 stored elements in Compressed Sparse Row format>
    """
    if logger is not None:
        logger.info("Start CPM normalization")

    # Calcualte diag matrix of the reciprocals of the row sums
    cell_counts = diags(1 / sp_matrix.sum(axis=1).A.ravel())
    # Multiply with scaling factor
    multplied_counts = sp_matrix.dot(_CPM_SCALING_FACT)

    # Caclualte the CPM normalized values
    cpm_normalized = cell_counts.dot(multplied_counts)

    if logger is not None:
        logger.info("Finished CPM normalization")

    return cpm_normalized


def sparse_log(
    sparse_matrix: csr_matrix, logger: Optional[Logger] = None
) -> csr_matrix:
    """
    Apply a natural logarithm transformation to a sparse count matrix.

    This function calculates the natural log transformation for a sparse count matrix,
    adding 1 to each value to handle zero counts.

    Args:
        sparse_matrix (csr_matrix): Sparse count matrix (CSR format).
        logger (Optional[Logger]): Logger for progress messages. Defaults to None.

    Returns:
        csr_matrix: Log-transformed sparse count matrix.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> count_matrix = csr_matrix([[0, 2, 3], [4, 0, 6]])
        >>> sparse_log(count_matrix)
        <2x3 sparse matrix of type '<class 'numpy.float64'>'
            with 6 stored elements in Compressed Sparse Row format>
    """
    if logger is not None:
        logger.info("Start log transformation")

    # Add +1 to each count and calculate natural log
    log_transformed = sparse_matrix.log1p()

    if logger is not None:
        logger.info("Finished log transformation")

    return log_transformed


def sparse_min_max(
    sp_matrix: csr_matrix,
    min_val: int = 0,
    max_val: int = 1,
    logger: Optional[Logger] = None,
) -> csr_matrix:
    """
    Rescale a sparse count matrix to a specified range using min-max scaling.

    This function rescales the values for each feature (gene) in a sparse count matrix to the specified range.

    Args:
        sp_matrix (csr_matrix): Sparse count matrix (CSR format).
        min_val (float, optional): Minimum value after rescaling. Defaults to 0.
        max_val (float, optional): Maximum value after rescaling. Defaults to 1.
        logger (Optional[Logger]): Logger for progress messages. Defaults to None.

    Returns:
        csr_matrix: Min-max rescaled sparse count matrix.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> count_matrix = csr_matrix([[1, 2, 3], [4, 5, 6]])
        >>> sparse_min_max(count_matrix, 0, 1)
        <2x3 sparse matrix of type '<class 'numpy.float64'>'
            with 6 stored elements in Compressed Sparse Row format>
    """
    if logger is not None:
        logger.info("Start log transformation")

    # Change matrix type so value are save correctly
    if sp_matrix.dtype != float:
        sp_matrix = sp_matrix.astype(float)

    # Convert to Compressed Sparse Column (CSC) format
    # -> Improves computation
    sp_matrix = sp_matrix.tocsc()

    # Iterate over all features
    for idx in tqdm(range(sp_matrix.shape[1]), desc="Normalize features"):
        feature_data = sp_matrix[:, idx]
        # Calculate normalized values for feature
        sp_matrix[:, idx] = (feature_data - feature_data.min()) / (
            feature_data.max() - feature_data.min() + _EPSILON
        ) * (max_val - min_val) + min_val

    # Convert back to CSR format
    min_max_scaled = sp_matrix.tocsr()

    return min_max_scaled