"""
This module provides functionality to load, normalize, and save single-cell RNA sequencing data.

The module uses the `anndata` library to handle HDF5 formatted data and performs normalization
using custom functions from a self-built `normalization` module. The command line interface
allows users to specify the data file to process.

Functions:
    load_data(file_path: str) -> ad.AnnData:
        Loads the AnnData object from the specified file.

    save_layer(adata: ad.AnnData, normalized_data: csr_matrix, filename: str) -> None:
        Saves the normalized data as a layer in the AnnData object and writes it to a file.

    main() -> None:
        Main function to execute the data loading, normalization, and saving based on command line arguments.

Example:
    To run the script from the command line:
        $ python script_name.py path/to/data.h5ad
"""

# Standard imports
import argparse
import os

# Third-party imports
import anndata as ad
from scipy.sparse import csr_matrix

# Self-built modules
from modules.logging_setup import main_logger
import modules.normalization as normalization

__author__ = "Christian Kolland"
__version__ = 0.1


def load_data(file_path: str) -> ad.AnnData:
    """Loads the AnnData object from a specified file.

    Args:
        file_path (str): Path to the H5AD file containing the AnnData object.

    Returns:
        ad.AnnData: The loaded AnnData object.

    Example:
        >>> adata = load_data("path/to/data.h5ad")
    """
    # Load AnnData object
    adata = ad.read_h5ad(filename=file_path)

    # Logging
    main_logger.info(f"Loaded dataset: {os.path.basename(file_path)}")

    return adata


def save_layer(adata: ad.AnnData, normalized_data: csr_matrix, filename: str) -> None:
    """Saves the normalized data as a layer in the AnnData object and writes it to a file.

    Args:
        adata (ad.AnnData): The AnnData object to update.
        normalized_data (csr_matrix): The normalized data to add as a layer.
        filename (str): The filename to write the updated AnnData object to.

    Example:
        >>> save_layer(adata, normalized_data, "normalized_data.h5ad")
    """
    # Add layer information
    adata.layers["min_max_normalized"] = normalized_data

    # Write updated AnnData object to file
    adata.write_h5ad(filename=filename)

    # Logging
    main_logger.info(f"Saved normalized data")


def main() -> None:
    """Main function to load, normalize, and save single-cell RNA sequencing data.

    This function parses command line arguments to get the input data file path,
    performs normalization, and saves the normalized data.

    Example:
        To run the script from the command line:
            $ python script_name.py path/to/data.h5ad
    """
    ## Command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    # Parse arguments
    args = parser.parse_args()

    ## Load data
    adata = load_data(file_path=args.data)

    ## Started CPM normalization
    # Logging
    main_logger.info(f"Start CPM normalization")

    cpm_normalized = normalization.sparse_cpm(adata.X)

    # Logging
    main_logger.info(f"Finished CPM normalization")

    ## Log transformation
    # Logging
    main_logger.info(f"Start Log transformation")

    log_transformed = cpm_normalized.log1p()

    # Logging
    main_logger.info(f"Finished Log transformation")

    ## Min-max normalization
    # Logging
    main_logger.info(f"Start Min-Max normalization")

    min_max_normalized = normalization.sparse_min_max(log_transformed)

    # Logging
    main_logger.info(f"Finished Min-Max normalization")

    ## Save data
    # Use the base name of the input file with a modified suffix or a different file name
    filename = os.path.splitext(os.path.basename(args.data))[0] + "_normalized.h5ad"
    save_layer(adata, normalized_data=min_max_normalized, filename=filename)


if __name__ == "__main__":
    main()
