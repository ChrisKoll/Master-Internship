# == Third-party ==
from anndata import AnnData
import numpy as np

# == Local ==
import constants as const


class Normalizer:
    """
    Provides different normalization methods.
    """

    def __init__(self, adata: AnnData):
        """ Constructor

        :param adata: AnnData Object
        """
        self.adata = adata
        self.cdata = self.adata.X.toarray()

    def update_cdata(self, new_cdata: np.ndarray):
        """
        Updates the cdata attribute.

        :param new_cdata: New count data
        """
        self.cdata = new_cdata

    def count_per_million_normalization(self) -> np.ndarray:
        """
        Count per Million normalization.
        """
        # Calculate total counts for each sample
        total_counts = self.cdata.sum()

        # Calculate scaling factor (counts per million)
        scaling_factor = const.CPM_SCALING_FACT

        # Perform CPM normalization
        cpm_normalized_data = (self.cdata / total_counts) * scaling_factor

        return cpm_normalized_data

    def median_of_ratios(self) -> np.ndarray:
        """
        Median of Ratios normalization.
        """
        # Calculate the geometric mean of counts across all samples for each gene
        geometric_means = np.exp(np.mean(np.log(self.cdata), axis=1))

        # Calculate the ratios of counts to geometric means
        ratios = self.cdata / geometric_means[:, np.newaxis]

        # Calculate the median of ratios across all genes
        median_ratios = np.median(ratios, axis=0)

        # Normalize the count data by dividing by the median of ratios
        normalized_data = self.cdata / median_ratios

        return normalized_data

    def min_max_normalization(self):
        """
        Min-Max normalization.
        """
        # Calculate the minimum and maximum values for each feature
        min_values = np.min(self.cdata, axis=0)
        max_values = np.max(self.cdata, axis=0)

        # Perform Min-Max normalization
        normalized_data = (self.cdata - min_values) / (max_values - min_values)

        return normalized_data

