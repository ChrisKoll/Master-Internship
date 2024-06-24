"""
Custom dataset class for sparse data.
Inherits from Pytorchs class 'Dataset'.
"""

# === Libraries ===
import torch
from torch.utils.data import Dataset


# === Class ===
class SparseDataset(Dataset):
    """
    Custom dataset class for sparse data.
    """

    def __init__(self, sparse_data):
        self.sparse_data = sparse_data

    def __len__(self):
        return self.sparse_data.shape[0]

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("Index out of range")

        # Extract the row as a dense numpy array
        row = self.sparse_data.getrow(index).toarray().squeeze()

        # Convert the row to a PyTorch tensor
        return torch.tensor(row, dtype=torch.float32)
