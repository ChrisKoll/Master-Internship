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


def sparse_cpm(sp_matrix: csr_matrix) -> np.array:
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
