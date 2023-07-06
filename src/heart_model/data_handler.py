# Standard library imports
from os import path
from pathlib import Path
from typing import Optional

# Third-party library imports
from anndata import AnnData
from scanpy import read_h5ad
from torch import torch


class Handler:
    """
    Class for handling h5ad data.
    """

    def __init__(self, file_location: Optional[str] = None):
        """Constructor

        :param file_location: Path to h5ad file
        """
        self.file_location = file_location
        self.file_name: Optional[str] = None
        self.adata = self.read_data()

    def read_data(self) -> AnnData:
        """
        Uses the scanpy function read_h5ad to import the anndata object.

        :return: Anndata object
        """
        if not path.exists(self.file_location) or self.file_location is None:
            raise ValueError("Invalid or no file path provided.")
        else:
            self.file_name = self.file_location.split("\\")[-1].split(".")[0]
            annotated_data = read_h5ad(filename=self.file_location)

            return annotated_data

    def subset_adata(self, export_path: str = None, *, number_rows: int = 5000, number_cols: int = 5000):
        """
        Subsets a given anndata object to a given size.

        :param export_path: Path for the file to be exported
        :param number_rows: Number of rows kept in the subset
        :param number_cols: Number of columns kept in the subset
        """
        if export_path is None:
            export_path = f"{self.file_name}_{number_rows}x{number_cols}_sample.h5ad"
        subset = self.adata[:number_cols, :number_rows]
        subset.write_h5ad(Path(export_path))

    def get_donors(self) -> list[str]:
        """
        Returns all sample donors.

        :return: Set of donors
        """
        donors = list(set(self.adata.obs["donor"]))

        return donors

    def to_tensor(self) -> torch.Tensor:
        """
        Converts the anndata expression matrix to a tensor.

        :return: Anndata tensor
        """
        tensor_data = torch.tensor(self.adata.X)

        return tensor_data
