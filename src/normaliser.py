# Third-party library imports
import numpy as np


class Normalizer:
    """
    Docstring
    """

    def __int__(self, count_data: np.ndarray):
        """ Constructor

        :param count_data: Expression matrix
        """
        self.count_data = count_data

    def count_per_million_normalization(self) -> np.ndarray:
        """
        Docstring
        """
        # Calculate total counts for each sample
        total_counts = self.count_data.sum()

        # Calculate scaling factor (counts per million)
        scaling_factor = 1e6

        # Perform CPM normalization
        cpm_normalized_data = (self.count_data / total_counts) * scaling_factor

        return cpm_normalized_data

    def median_of_ratios(self) -> np.ndarray:
        """
        Docstring
        """
        # Calculate the geometric mean of counts across all samples for each gene
        geometric_means = np.exp(np.mean(np.log(self.count_data), axis=1))

        # Calculate the ratios of counts to geometric means
        ratios = self.count_data / geometric_means[:, np.newaxis]

        # Calculate the median of ratios across all genes
        median_ratios = np.median(ratios, axis=0)

        # Normalize the count data by dividing by the median of ratios
        normalized_data = self.count_data / median_ratios

        return normalized_data

    def min_max_normalization(self):
        """
        Docstring
        """
        # Calculate the minimum and maximum values for each feature
        min_values = np.min(self.count_data, axis=0)
        max_values = np.max(self.count_data, axis=0)

        # Perform Min-Max normalization
        normalized_data = (self.count_data - min_values) / (max_values - min_values)

        return normalized_data

