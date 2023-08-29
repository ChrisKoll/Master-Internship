# == Standard ==
from os import path
from pathlib import Path
from typing import Optional

# == Third-party ==
from anndata import AnnData
import numpy as np
from scanpy import read_h5ad

# == Local imports ==
from helpers.analyst import Analyst


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
        self.count_data: Optional[np.matrix] = None
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
            self.count_data = np.array(annotated_data.X)

            return annotated_data

    def subset_adata(self, export_path: Optional[str], *,
                     number_rows: Optional[int],
                     number_cols: Optional[int],
                     shuffle: bool = True):
        """
        Subsets a given anndata object to a given size.

        :param export_path: Path for the file to be exported
        :param number_rows: Number of rows kept in the subset
        :param number_cols: Number of columns kept in the subset
        :param shuffle: Decides if the rows in the matrix are shuffled
        """
        if number_rows is None:
            number_rows = 5000
        if number_cols is None:
            number_cols = 5000
        if export_path is None:
            export_path = f"{self.file_name}_{number_rows}x{number_cols}_sample.h5ad"

        # Subset data
        if shuffle is True:
            adata = self.adata
            # Add the shuffling
            subset = adata[:number_cols, :number_rows]
            subset.write_h5ad(Path(export_path))
        else:
            subset = self.adata[:number_cols, :number_rows]
            subset.write_h5ad(Path(export_path))

    def data_analysis(self, statistical: bool = False, pca: bool = False, svd: bool = False):
        """
        Start different analysis methods for data analysis.

        :param statistical: Starts the statistical analysis
        :param pca: Starts the pca analysis
        :param svd: Starts the svd analysis
        """
        # Create Analyst object (analyst.py)
        analyst = Analyst(self.adata)

        # Start the analysis according to the passed arguments
        if statistical is True:
            analyst.statistical_analysis()
        if pca is True:
            analyst.pca_analysis()
        if svd is True:
            analyst.svd_analysis()


if __name__ == '__main__':
    file = "/home/ubuntu/Projects/Master-Internship/data/global_raw_5000x5000_sample.h5ad"
    handler = Handler(file_location=file)
    handler.data_analysis(True, True, True)
