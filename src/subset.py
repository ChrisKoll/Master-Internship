from pathlib import Path
from random import choice
from typing import Optional

import anndata
import numpy as np


def subset_adata(
    adata: anndata.AnnData,
    export_path: Optional[str] = None,
    *,
    number_rows: Optional[int],
    number_cols: Optional[int],
):
    """
    Subsets a given anndata object to a given size.

    :param export_path: Path for the file to be exported
    :param number_rows: Number of rows kept in the subset
    :param number_cols: Number of columns kept in the subset
    """
    if number_rows is None:
        number_rows = 5000
    if number_cols is None:
        number_cols = 5000
    if export_path is None:
        export_path = f"adata_{number_rows}x{number_cols}_sample.h5ad"

    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices]

    subset = adata[:number_rows, :number_cols]
    subset.write_h5ad(Path(export_path))


if __name__ == "__main__":
    # file = "data/global_raw.h5ad"
    file = "data/adata_20000x10000_sample.h5ad"
    adata = anndata.read_h5ad(file)
    # subset_adata(adata, number_rows=20000, number_cols=10000)
    print(choice(adata.obs["donor"].unique()))
