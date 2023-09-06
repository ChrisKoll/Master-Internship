# == Standard ==
from datetime import datetime
from os import getcwd
from shutil import move

# == Third-party ==
from anndata import AnnData
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA

# == Local ==
import constants as const


class Analyst:
    """
    Performs various data analysis methods.
    """

    def __init__(self, adata: AnnData):
        """Constructor

        :param adata: AnnData object
        """
        self.adata = adata

    def statistical_analysis(self):
        """
        Plots a number of statistical features.
        """
        # Creates pie plot
        # --> Amount of samples per donor
        self.plot_donor_distribution()

        # Creates pie plot
        # --> Amount of samples per cell type
        self.plot_cell_type_distribution()

        # Creates pie plot
        # --> Expression distribution for all 0 genes
        # self.plot_0_expression()

    def plot_donor_distribution(self):
        """
        Plots the amount of samples per donor in the AnnData object.
        """
        # List of unique donors
        # --> Act as labels for the plot
        labels = self.adata.obs["donor"].unique()

        # Returns the amount of samples the condition fits
        # --> Act as sizes for the plot
        sizes = []
        for donor in labels:
            sizes.append(self.adata[self.adata.obs["donor"] == donor].shape[0])

        # Calculate percentages for visualization
        percentages = self.calculate_percentages(sizes=sizes)

        self.plot_pie(labels=labels, sizes=sizes, percentages=percentages, title=const.PLOT_TITLE_DONOR_DIST)

    def plot_cell_type_distribution(self):
        """
        Plots the amount of samples per cell type in the AnnData object.
        """
        # List of unique cell types
        # --> Act as labels for the plot
        labels = self.adata.obs["cell_type"].unique()

        # Returns the amount of samples the condition fits
        # --> Act as sizes for the plot
        sizes = []
        for cell_type in labels:
            sizes.append(self.adata[self.adata.obs[const.OBS_CELL_TYPE] == cell_type].shape[0])

        # Calculate percentages for visualization
        percentages = self.calculate_percentages(sizes=sizes)

        self.plot_pie(labels=labels, sizes=sizes, percentages=percentages, title=const.PLOT_TITLE_CELL_TYPE_DIST)

    def calculate_percentages(self, sizes: list[int]) -> list[float]:
        """
        Calculates percentages.
        """
        # Get total amount of samples
        total = self.adata.obs_names.shape[0]

        percentages = [((size / total) * 100) for size in sizes]

        return percentages

    def plot_0_expression(self):
        """
        Plots the distribution of genes that have a 0 expression over all samples.
        """
        # Genes with 0 expression for all samples
        zero_exp = self.bdata.columns[(self.bdata == 0).all()].tolist()

        # Other genes
        not_zero_exp = [gene for gene in self.adata.var_names if gene not in zero_exp]

        # Calculate plotting parameters
        labels = [const.LABEL1_0_EXP, const.LABEL2_0_EXP]
        sizes = [len(zero_exp), len(not_zero_exp)]
        # Calculate percentages
        percentages = [(len(zero_exp) / len(self.adata.obs_names) * 100),
                       (len(not_zero_exp) / len(self.adata.obs_names) * 100)]

        self.plot_pie(labels=labels, sizes=sizes, percentages=percentages, title=const.PLOT_TITLE_0_EXP)

    def pca_analysis(self):
        """
        Performs a PCA analysis for the provided data.
        """
        # Filters indices (genes) with non-zero expression
        filtered_counts = self.bdata.loc[:, (self.bdata != 0).any(axis=0)]

        # Generate the PCA components
        pca = PCA(n_components=const.NUMBER_COMPONENTS)
        data_pca = pca.fit_transform(filtered_counts)

        self.plot_results(data_pca, analysis="pca")

    def svd_analysis(self):
        """
        Performs an SVD analysis for the provided data.
        """
        # Generate the SVD components
        svd = TruncatedSVD(n_components=const.NUMBER_COMPONENTS)
        data_svd = svd.fit_transform(self.adata.X)

        self.plot_results(data_svd, analysis="svd")

    def plot_pie(self, labels: list[str], sizes: list[int], percentages: list[float], title: str):
        """
        Creates a pie chart.

        :param labels: Slice labels for the chart
        :param sizes: Slice sizes for the chart
        :param percentages: Percentages of sizes
        :param title: Title of the chart
        """
        fig, ax = plt.subplots()

        # Explode values
        explode = [0.05] * len(sizes)

        # Pie chart parameters
        patches, texts = ax.pie(x=sizes, explode=explode, startangle=90, pctdistance=0.85)
        labels = [f'{label} ({perc:0.1f}%)' for label, perc in zip(labels, percentages)]
        ax.legend(patches, labels, loc="best")
        ax.set_title(title)

        # Creating center circle
        center_circle = plt.Circle(xy=(0, 0), radius=0.7, fc="white")
        fig = plt.gcf()
        # Adding circle
        fig.gca().add_artist(center_circle)
        ax.axis("equal")

        # Save the pie chart as an image (PNG format)
        now = datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f')
        plt.savefig(f"{now}.png")

        self.move_plot(name=now)

    def plot_results(self, data, *, analysis: str):
        """
        Plots the results of the pca / svd analysis.

        :param data: Component analysis data
        :param analysis: Analysis method
        """
        # Combine the components for plotting
        to_plot = []
        for idx in range(const.NUMBER_COMPONENTS - 1):
            # For each iteration combine:
            # Component of current iteration + next component
            to_plot.append((data[:, idx], data[:, idx + 1]))

        # Iterates over the plotting data
        for idx, plot in enumerate(to_plot):
            # Resets plot
            plt.figure()

            # Generates the plot
            plt.scatter(plot[0], plot[1])
            plt.xlabel(f"Component {idx + 1}")
            plt.ylabel(f"Component {idx + 2}")
            plt.title(f"{analysis.upper()} Analysis")

            # Save the pie chart as an image (PNG format)
            now = datetime.now().strftime('%d-%m-%Y_%H:%M:%S:%f')
            plt.savefig(f"{now}.png")

            self.move_plot(name=now)

    @staticmethod
    def move_plot(name: str):
        """
        Ḿoves a given file to the tmp folder.

        :param name: Name of the file
        """
        move(src=f"{name}.png", dst=f"{getcwd()}/{const.FILE_DESTINATION}")
