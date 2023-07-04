# Standard library imports
from os import path
from typing import Optional

# Local/application-specific import
import constants as c

# Third-party library imports
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scanpy import read_h5ad
from sklearn.decomposition import TruncatedSVD


class Analyzer:

    def __init__(self, file_location: Optional[str] = None):
        """Constructor

        :param file_location: Path to h5ad file
        """
        self.file_location = file_location
        self.file_name: Optional[str] = None
        self.adata: Optional[AnnData] = None

    def read_file(self):
        """
        Reads the h5ad file.
        """
        if not path.exists(self.file_location) or self.file_location is None:
            raise ValueError("Invalid or no file path provided.")
        else:
            self.file_name = self.file_location.split("\\")[-1]
            self.adata = read_h5ad(filename=self.file_location)

    def save_plots(self, plots, *, analysis: str):
        """
        Saves a number of given plots to pdf.

        :param plots: Plots that are saved
        :param analysis: Type of analysis
        """
        # Creates PDF object
        pdf = PdfPages("_".join([self.file_name, analysis]))

        # Iterate over the plots and save each one to the PDF file
        for plot in plots:
            # Create a new figure for each plot
            plt.figure()

            # Generate the plot (replace this with your own plotting code)
            plt.plot(plot['x'], plot['y'])
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(plot['title'])

            # Save the current figure to the PDF file
            pdf.savefig()

        # Close the PDF file
        pdf.close()

    def pca_analysis(self):
        pass

    def svd_analysis(self):
        svd = TruncatedSVD(n_components=c.NUMBER_COMPONENTS)
        data_svd = svd.fit_transform(self.adata)

        plots = []
        for idx, component in enumerate(data_svd):
            plots.append()
