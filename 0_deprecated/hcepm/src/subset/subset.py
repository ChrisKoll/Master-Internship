"""
SUBSET module:
Contains functionality to subset a given AnnData object.
The size of the subset can be decided by the user.
"""

# Third-party libs
import anndata
import numpy as np


def subset_adata(
    adata: anndata.AnnData,
    *,
    number_rows: int,
    number_cols: int,
    shuffle: bool = True,
) -> anndata.AnnData:
    """
    Subsets a given anndata object to a given size.

    :param export_path: Path for the file to be exported
    :param number_rows: Number of rows kept in the subset (default: all rows)
    :param number_cols: Number of columns kept in the subset (default: all columns)
    """
    if shuffle:
        # Shuffle indices before subsetting
        shuffled_indices = np.random.permutation(adata.n_obs)
        adata = adata[shuffled_indices]

        return adata[:number_rows, :number_cols]
    else:
        # Indices are not shuffled
        return adata[:number_rows, :number_cols]
