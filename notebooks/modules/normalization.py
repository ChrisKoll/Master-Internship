"""
Collection of normalization methods used in 'normalization.ipynb'
"""

# === Libraries ===
import numpy as np
from scipy.sparse import csr_matrix, diags

# === Constants ===
CPM_SCALING_FACT = 1e6

# === Functions ===


def dense_cpm(count_matrix: np.array) -> np.array:
    """
    Calculates the Counts Per Million (CPM) normalization for a count matrix.
    Count matrix format: rows = samples/cells, cols = genes

    Args:
        count_matrix (numpy.array): Count matrix.

    Returns:
        np.array: Normalized counts.
    """
    # Sums over all rows to get total counts
    cell_counts = np.sum(count_matrix, axis=1)

    # Caclualte the CPM normalized values
    cpm_matrix = count_matrix * CPM_SCALING_FACT / cell_counts[:, np.newaxis]

    return cpm_matrix


def sparse_cpm(sp_matrix: csr_matrix) -> csr_matrix:
    """
    Calculates the Counts Per Million (CPM) normalization for a count matrix in the sparse format.
    Count matrix format: rows = samples/cells, cols = genes

    Args:
        sp_matrix (csr_matrix): Count matrix (Compressed Sparse Row format).

    Returns:
        csr_matrix: Normalized counts.
    """
    # Calcualte diag matrix of the reciprocals of the row sums
    cell_counts = diags(1 / sp_matrix.sum(axis=1).A.ravel())
    # Multiply with scaling factor
    multplied_counts = sp_matrix.dot(CPM_SCALING_FACT)

    # Caclualte the CPM normalized values
    cpm_matrix = cell_counts.dot(multplied_counts)

    return cpm_matrix


def dense_log(count_matrix: np.array) -> np.array:
    """
    Calculates natural log transformation for a count matrix.
    Adds 1 to each value to produce a numeric result for 0 counts.
    Count matrix format: rows = samples/cells, cols = genes

    Args:
        count_matrix (numpy.array): Count matrix.

    Returns:
        np.array: Transformed counts.
    """
    # Calculate log values
    # --> +1 to each value
    log_matrix = np.log(count_matrix + 1)

    return log_matrix


def dense_min_max(
    count_matrix: np.array, min_val: int = 0, max_val: int = 1
) -> np.array:
    """
    Rescales the values from a count matrix to the given range.
    Default boundaries are [0, 1].
    Count matrix format: rows = samples/cells, cols = genes

    Args:
        count_matrix (numpy.array): Count matrix.
        min_val (int): Minimal value after rescaling.
        max_val (int): Maximum value after rescaling.

    Returns:
        np.array: Rescaled counts.
    """
    # Determines min value in data
    min_data = np.min(count_matrix)
    # Determines max value in data
    max_data = np.max(count_matrix)

    # Rescales values to given range
    min_max_normalized = (count_matrix - min_data) / (max_data - min_data) * (
        max_val - min_val
    ) + min_val

    return min_max_normalized


def sparse_min_max(
    sp_matrix: csr_matrix, min_val: int = 0, max_val: int = 1
) -> csr_matrix:
    # Get min value from sparse matrix
    min_data = sp_matrix.min()
    # Get max value from sparse matrix
    max_data = sp_matrix.max()

    # Calculate Min-Max as described above
    min_max_matrix = (sp_matrix - min_data) / (max_data - min_data) * (
        max_val - min_val
    ) + min_val

    return min_max_matrix
