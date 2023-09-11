# == Standard ==
from datetime import datetime
from os import getcwd
from shutil import move
from typing import Optional

# == Third-party ==
from anndata import AnnData
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA

# == Local ==
import constants as const


# |===================|
# |-- Helpers --|
# |===================|


# |===================|
# |-- Data Analysis --|
# |===================|


def plot_donor_distribution(adata: AnnData):
    """
    Plots the amount of samples per donor in the AnnData object.

    :param adata: AnnData object
    """
    # List of unique donors
    # --> Act as labels for the plot
    labels = adata.obs["donor"].unique().tolist()

    # Returns the amount of samples the condition fits
    # --> Act as sizes for the plot
    sizes = []
    for donor in labels:
        sizes.append(adata[adata.obs["donor"] == donor].shape[0])

    # Calculate percentages for visualization
    percentages = calculate_percentages(adata=adata, sizes=sizes, base="samples")

    if percentages is not None:
        plot_pie(
            labels=labels,
            sizes=sizes,
            percentages=percentages,
            title=const.PLOT_TITLE_DONOR_DIST,
        )


def plot_cell_type_distribution(adata: AnnData):
    """
    Plots the amount of samples per cell type in the AnnData object.

    :param adata: AnnData object
    """
    # List of unique cell types
    # --> Act as labels for the plot
    labels = adata.obs["cell_type"].unique().tolist()

    # Returns the amount of samples the condition fits
    # --> Act as sizes for the plot
    sizes = []
    for cell_type in labels:
        sizes.append(adata[adata.obs[const.OBS_CELL_TYPE] == cell_type].shape[0])

    # Calculate percentages for visualization
    percentages = calculate_percentages(adata=adata, sizes=sizes, base="samples")

    if percentages is not None:
        plot_pie(
            labels=labels,
            sizes=sizes,
            percentages=percentages,
            title=const.PLOT_TITLE_CELL_TYPE_DIST,
        )


def plot_0_expression(adata: AnnData):
    """
    Plots the distribution of genes that have a 0 expression over all samples.

    :param adata: AnnData object
    """
    # Convert Compressed Sparse Row (csr) matrix to csc
    bdata = sparse.csc_matrix(adata.X)

    # Count all genes with 0 expression
    # --> Sum over column values
    zero_exp = 0
    for col_idx in range(len(adata.var_names)):
        column = bdata.getcol(col_idx)
        if column.sum() == 0:
            zero_exp += 1

    # Genes with not all zero expression
    not_zero_exp = len(adata.var_names) - zero_exp

    # Calculate plotting parameters
    labels = [const.LABEL1_0_EXP, const.LABEL2_0_EXP]
    sizes = [zero_exp, not_zero_exp]

    # Calculate percentages for visualization
    percentages = calculate_percentages(adata=adata, sizes=sizes, base="genes")

    if percentages is not None:
        plot_pie(
            labels=labels,
            sizes=sizes,
            percentages=percentages,
            title=const.PLOT_TITLE_0_EXP,
        )


def calculate_percentages(
    adata: AnnData, sizes: list[int], base: str
) -> Optional[list[float]]:
    """
    Calculates percentages.

    :param adata: AnnData object
    :param sizes: Sizes that are used for percentage calculation
    """
    if base in const.ARGUMENT_COLLECTION_PERCENTAGES:
        if base == "samples":
            # Get total amount of samples
            total = adata.obs_names.shape[0]

            percentages = [((size / total) * 100) for size in sizes]

            return percentages

        elif base == "genes":
            # Get total amount of genes
            total = adata.var_names.shape[0]

            percentages = [((size / total) * 100) for size in sizes]

            return percentages
    else:
        raise ValueError(
            f"calculate_percentages: base must be one of {const.ARGUMENT_COLLECTION_PERCENTAGES}"
        )


def plot_pie(labels: list[str], sizes: list[int], percentages: list[float], title: str):
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
    patches = ax.pie(x=sizes, explode=explode, startangle=90, pctdistance=0.85)
    labels = [f"{label} ({perc:0.1f}%)" for label, perc in zip(labels, percentages)]
    ax.legend(patches[0], labels, loc="best")
    ax.set_title(title)

    # Creating center circle
    center_circle = Circle(xy=(0, 0), radius=0.7, fc="white")
    fig = plt.gcf()
    # Adding circle
    fig.gca().add_artist(center_circle)
    ax.axis("equal")

    # Save the pie chart as an image (PNG format)
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S:%f")
    plt.savefig(f"{now}.png")


# |========================|
# |-- Data Normalization --|
# |========================|


def count_per_million_normalization(cdata):
    """
    Count per Million normalization.

    :param cdata: Sparse matrix
    """
    # Calculate total counts for each sample
    total_counts = np.sum(cdata, axis=1)

    # Calculate scaling factor (counts per million)
    scaling_factor = const.CPM_SCALING_FACT

    # Perform CPM normalization
    cpm_normalized = (cdata / total_counts) * scaling_factor

    return cpm_normalized


def median_of_ratios(cdata):
    """
    Median of Ratios normalization.

    :param cdata: Sparse matrix
    """
    # Calculate the geometric mean of counts across all samples for each gene
    geometric_means = np.exp(np.mean(np.log(cdata), axis=1))

    # Calculate the ratios of counts to geometric means
    ratios = cdata / geometric_means[:, np.newaxis]

    # Calculate the median of ratios across all genes
    median_ratios = np.median(ratios, axis=0)

    # Normalize the count data by dividing by the median of ratios
    mor_normalized = cdata / median_ratios

    return mor_normalized


def min_max_normalization(cdata):
    """
    Min-Max normalization.

    :param cdata: Sparse matrix
    """
    # Calculate the minimum and maximum values for each feature
    data_min = np.min(cdata)
    data_max = np.max(cdata)

    # Perform Min-Max normalization
    min_max_normalized = (cdata - data_min) / (data_max - data_min)

    return min_max_normalized
