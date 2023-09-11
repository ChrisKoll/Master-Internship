# == Testing ==
import numpy as np
from scipy import sparse

# == Standard ==
from os import path

# == Third-party ==
from anndata import AnnData, read_h5ad

# == Local imports ==
import util


class Handler:
    """
    Class for handling h5ad data.
    """

    def __init__(self, file_location: str):
        """Constructor

        :param file_location: Path to h5ad file
        """
        self.file_location = file_location
        self.adata: AnnData

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
            self.adata = read_h5ad(filename=self.file_location)

    def statistical_analysis(self):
        """
        Plots a number of statistical features.
        """
        # Creates pie plot
        # --> Amount of samples per donor
        util.plot_donor_distribution(adata=self.adata)

        # Creates pie plot
        # --> Amount of samples per cell type
        util.plot_cell_type_distribution(adata=self.adata)

        # Creates pie plot
        # --> Expression distribution for all 0 genes
        util.plot_0_expression(adata=self.adata)

    def normalize(self):
        """
        Normalizes the data after a given format.
        """
        # Count Per Million normalization
        self.adata.layers["cpm_normalized"] = util.count_per_million_normalization(
            self.adata.X
        )
        # Use Min-Max normalization to bring data in range of 0 - 1
        self.adata.layers["cpm_minmax_normalized"] = util.min_max_normalization(
            self.adata.layers["cpm_normalized"]
        )
