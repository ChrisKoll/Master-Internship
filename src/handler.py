# == Testing ==
import numpy as np
from scipy import sparse

# == Standard ==
from os import path
from typing import Optional

# == Third-party ==
from anndata import AnnData, read_h5ad

# == Local imports ==
from helpers.analyst import Analyst


class Handler:
    """
    Class for handling h5ad data.
    """

    def __init__(self, file_location: str):
        """Constructor

        :param file_location: Path to h5ad file
        """
        self.file_location = file_location
        self.adata: Optional[AnnData] = None

        # Read the h5ad file
        self.read_data()

    def read_data(self):
        """
        Uses the scanpy function read_h5ad to import the anndata object.

        :return: Anndata object
        """
        if not path.exists(self.file_location):
            raise ValueError("Invalid file path.")
        else:
            self.adata = read_h5ad(filename=self.file_location, backed='r+')

    def data_analysis(self):
        """
        Start different analysis methods for data analysis.
        """
        # Create Analyst object (analyst.py)
        analyst = Analyst(self.adata)
        analyst.statistical_analysis()

    def test_function(self):
        for _, idx in enumerate(self.adata.var_names):
            print(idx)
            test = self.adata[:10, idx].X
            print(test != 0.0)
            """
            if test is None:
                print("0er")
            else:
                print(idx)
            """
        # print(self.adata[0, 0].X)
        # zero_columns_idx = np.where(np.all(self.adata.X == 0.0, axis=0))
        # print(zero_columns_idx)


if __name__ == '__main__':
    # file = "/media/sf_Share/Schulzlab/global_raw.h5ad"
    file = "/home/ubuntu/Projects/Master-Internship/data/global_raw_5000x5000_sample.h5ad"
    handler = Handler(file_location=file)
    handler.test_function()
    # handler.data_analysis()
    # sparse_matrix = sparse.csr_matrix(handler.adata.X)


