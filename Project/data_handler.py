# Standard library imports
from os import path
from pathlib import Path
from typing import Optional

# Third-party library imports
from anndata import AnnData
from scanpy import read_h5ad
from torch import torch


class DataHandler:
    """
    Class for handling h5ad data.
    """

    def __init__(self, file_location: Optional[str] = None):
        """Constructor

        :param file_location: Path to h5ad file
        """
        self.file_location = file_location
        self.adata = self.read_data()

    def read_data(self) -> AnnData:
        """
        Uses the scanpy function read_h5ad to import the anndata object.

        :return: Anndata object
        """
        if path.exists(self.file_location) or self.file_location is None:
            raise ValueError("Invalid or no file path provided.")

        annotated_data = read_h5ad(filename=self.file_location)

        return annotated_data

    def subset_adata(self, export_path: str, *, number_cols: int = 5000, number_rows: int = 5000):
        """
        Subsets a given anndata object to a given size.

        :param export_path: Path for the file to be exported
        :param number_cols: Number of columns kept in the subset
        :param number_rows: Number of rows kept in the subset
        """
        subset = self.adata[:number_cols, :number_rows]
        subset.write_h5ad(Path(export_path))

    def to_tensor(self) -> torch.Tensor:
        """
        Converts the anndata expression matrix to a tensor.

        :return: Anndata tensor
        """
        expression_data = self.adata.X
        tensor_data = torch.tensor(expression_data)

        return tensor_data
