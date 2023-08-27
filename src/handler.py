# Standard library imports
from os import path
from pathlib import Path
from typing import Optional

# Third-party library imports
from anndata import AnnData
import numpy as np
from scanpy import read_h5ad
from torch import torch

import src.analyst


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
        self.expression_matrix: Optional[np.matrix] = None
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

            # Extract expression matrix
            self.expression_matrix = np.array(annotated_data.X)
            print(self.expression_matrix)

            return annotated_data

    def subset_adata(self, export_path: Optional[str] = None, *,
                     number_rows: int = 5000,
                     number_cols: int = 5000,
                     shuffle: bool = True):
        """
        Subsets a given anndata object to a given size.

        :param export_path: Path for the file to be exported
        :param number_rows: Number of rows kept in the subset
        :param number_cols: Number of columns kept in the subset
        :param shuffle: Decides if the rows in the matrix are shuffled
        """
        if export_path is None:
            export_path = f"{self.file_name}_{number_rows}x{number_cols}_sample.h5ad"
        if shuffle is True:
            adata = self.adata
            np.random.shuffle(adata)
            subset = adata[:number_cols, :number_rows]
            subset.write_h5ad(Path(export_path))
        else:
            subset = self.adata[:number_cols, :number_rows]
            subset.write_h5ad(Path(export_path))

    def data_analysis(self, *args):
        """
        Docstring
        """
        analyst = src.analyst.Analyst(self.adata)

        if "statistics" in args:
            analyst.plot_expression()

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
        tensor_data = torch.tensor(self.expression_matrix)

        return tensor_data
