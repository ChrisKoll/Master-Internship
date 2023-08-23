# Standard library imports
from os import path
from typing import Optional

# Third-party library imports
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scanpy import read_h5ad
from sklearn.decomposition import TruncatedSVD, PCA

# Local import
import src.heart_model.constants as c


class Analyzer:
    """
    Analyzes a given dataset with different statistical analysis methods.
    """

    def __init__(self, adata: AnnData):
        """Constructor

        :param adata: Path to h5ad file
        """
        self.adata = adata
        self.count_data = self.extract_count_data()

    def extract_count_data(self):
        """
        Docstring
        """
        if self.adata is not None:
            return self.adata.X
        else:
            return None

    def pca_analysis(self):
        """
        Performs an PCA analysis for the provided data.
        """
        # Filters indices (genes) with non-zero expression
        non_zero = np.where(np.any(self.count_data != 0, axis=0))[0]
        filtered_counts = self.count_data[:, non_zero]

        # Generate the PCA components
        pca = PCA(n_components=c.NUMBER_COMPONENTS)
        data_pca = pca.fit_transform(filtered_counts)

        # UI
        print(f"{c.NUMBER_COMPONENTS} PCA components have been generated...")

        self.save_plots(data_pca, analysis="pca")

    def svd_analysis(self):
        """
        Performs an SVD analysis for the provided data.
        """
        # Generate the SVD components
        svd = TruncatedSVD(n_components=c.NUMBER_COMPONENTS)
        data_svd = svd.fit_transform(self.adata.X)

        # UI
        print(f"{c.NUMBER_COMPONENTS} SVD components have been generated...")

        self.save_plots(data_svd, analysis="svd")

    @staticmethod
    def save_plots(data, *, analysis: str):
        """
        Saves a number of given plots to pdf.

        :param data: Component analysis data
        :param analysis: Analysis method
        """
        # Combine the components for plotting
        to_plot = []
        for idx in range(c.NUMBER_COMPONENTS - 1):
            # For each iteration combine:
            # Component of current iteration + Next component
            to_plot.append((data[:, idx], data[:, idx + 1]))

        # Creates PDF object
        pdf = PdfPages(f"{analysis}.pdf")

        # Iterates over the plotting data
        for idx, plot in enumerate(to_plot):
            # Create a new figure
            plt.figure()

            # Generates the plot
            plt.scatter(plot[0], plot[1])
            plt.xlabel(f"Component {idx + 1}")
            plt.ylabel(f"Component {idx + 2}")
            plt.title(f"{analysis.upper()} Analysis")

            # Save the current figure to the PDF file
            pdf.savefig()

        # Close the PDF file
        pdf.close()

    def extract_statistics(self):
        """
        Docstring
        """
        pass
