"""
Custom dataset class for sparse data.

This module defines a custom dataset class, `SparseDataset`, which inherits from
PyTorch's `Dataset` class. The dataset is designed to handle sparse data efficiently.

Example:
    >>> from scipy.sparse import csr_matrix
    >>> import torch
    >>> sparse_data = csr_matrix([[0, 1, 0], [4, 0, 6]])
    >>> dataset = SparseDataset(sparse_data)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    >>> for data in loader:
    >>>     print(data)
"""

# Third-party imports
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset

__author__ = "Christian Kolland"
__version__ = 1.0


class SparseDataset(Dataset):
    """
    Custom dataset class for sparse data.

    This class provides an interface for working with sparse data in PyTorch.
    It supports indexing and retrieval of rows from sparse matrices, converting
    them into dense PyTorch tensors.

    Attributes:
        sparse_data (csr_matrix): A SciPy sparse matrix (CSR format) containing the data.

    Args:
        sparse_data (csr_matrix): A sparse matrix in Compressed Sparse Row (CSR) format.
    """

    def __init__(self, sparse_data: csr_matrix):
        """
        Initializes the SparseDataset with sparse data.

        Args:
            sparse_data (csr_matrix): The sparse matrix data to be used by the dataset.
        """
        self.sparse_data = sparse_data

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of rows in the sparse matrix.
        """
        return self.sparse_data.shape[0]

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: A tensor containing the data for the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index >= len(self):
            raise IndexError("Index out of range")

        # Extract the row as a dense numpy array
        row = self.sparse_data.getrow(index).toarray().squeeze()

        # Convert the row to a PyTorch tensor
        return torch.tensor(row, dtype=torch.float32)
