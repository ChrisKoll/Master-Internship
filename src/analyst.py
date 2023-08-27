# Third-party library imports
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA

# Local import
import src.constants as const


class Analyst:
    """
    Analyzes a given dataset with different statistical analysis methods.
    """

    def __init__(self, adata: AnnData = None):
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
        pca = PCA(n_components=const.NUMBER_COMPONENTS)
        data_pca = pca.fit_transform(filtered_counts)

        # UI
        print(f"{const.NUMBER_COMPONENTS} PCA components have been generated...")

        self.save_plots(data_pca, analysis="pca")

    def svd_analysis(self):
        """
        Performs an SVD analysis for the provided data.
        """
        # Generate the SVD components
        svd = TruncatedSVD(n_components=const.NUMBER_COMPONENTS)
        data_svd = svd.fit_transform(self.adata.X)

        # UI
        print(f"{const.NUMBER_COMPONENTS} SVD components have been generated...")

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
        for idx in range(const.NUMBER_COMPONENTS - 1):
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

    def plot_expression(self):
        """
        Docstring
        """
        # Filter genes with 0 expression and count non-zero genes
        zero_expression = [gene for gene in self.adata.var_names if np.all(self.adata[:, gene].X == 0)]
        non_zero_expression = [gene for gene in self.adata.var_names if np.all(self.adata[:, gene].X != 0)]
        other_genes = [gene for gene in self.adata.var_names if gene not in zero_expression and
                       gene not in non_zero_expression]

        # Data for the pie chart
        pie_sizes = [len(non_zero_expression), len(other_genes), len(non_zero_expression)]
        pie_labels = ["0-Expression Genes", "Others", "Non-0-Expression Genes"]
        explode_zero = [0.1, 0, 0]
        explode_non_zero = [0, 0, 0.1]

        bar_sizes = [1] * len(non_zero_expression)
        bar_labels = non_zero_expression

        self.create_pie_chart(pie_sizes, pie_labels, explode_zero, bar_sizes, bar_labels)

    def create_pie_chart(self, pie_sizes, pie_labels, explode, bar_sizes, bar_labels):
        """
        Docstring
        """
        # make figure and assign axis objects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
        fig.subplots_adjust(wspace=0)

        # pie chart parameters
        # rotate so that first wedge is split by the x-axis
        angle = -180 * (pie_sizes[0] / self.adata.var_names)
        wedges, *_ = ax1.pie(pie_sizes, autopct='%1.1f%%', startangle=angle,
                             labels=pie_labels, explode=explode)

        bottom = 1
        width = .2

        # Adding from the top matches the legend.
        for j, (height, label) in enumerate(reversed([*zip(bar_sizes, bar_labels)])):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label, alpha=0.1 + 0.25 * j)
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax2.set_title('Age of approvers')
        ax2.legend()
        ax2.axis('off')
        ax2.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(bar_sizes)

        # draw top connecting line
        x_top = r * np.cos(np.pi / 180 * theta2) + center[0]
        y_top = r * np.sin(np.pi / 180 * theta2) + center[1]
        connection = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData, xyB=(x_top, y_top),
                                     coordsB=ax1.transData)
        connection.set_color([0, 0, 0])
        connection.set_linewidth(4)
        ax2.add_artist(connection)

        # draw bottom connecting line
        x_bot = r * np.cos(np.pi / 180 * theta1) + center[0]
        y_bot = r * np.sin(np.pi / 180 * theta1) + center[1]
        connection = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData, xyB=(x_bot, y_bot),
                                     coordsB=ax1.transData)
        connection.set_color([0, 0, 0])
        ax2.add_artist(connection)
        connection.set_linewidth(4)

        # Save the pie chart as an image (PNG format)
        plt.savefig("pie_chart.png")


if __name__ == '__main__':
    new = Analyst()
