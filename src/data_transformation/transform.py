"""
Data Transformation Module.

This module performs data transformation operations on single-cell data, including 
Counts Per Million (CPM) normalization, log transformation, and min-max scaling. 
The module also supports logging and allows users to specify input data, log file path, 
and log file name through command-line arguments.

Functions:
    - standard_transform: Orchestrates the data transformation process using command-line inputs.
"""

# Standard imports
import argparse
import os

# Self-built modules
import src.data_transformation.methods as mtd
import src.utils.io_utils as io_utils
from src.utils.logging_utils import setup_logger

__author__ = "Christian Kolland"
__version__ = 1.0

# Constants
_PARSER_DESC = "Perform data transformation on single-cell data."
_ARG_DATA_HELP = "Path to the input data file (e.g., .h5ad file) for processing."
_ARG_LOG_HELP = "Directory path where the log file will be saved. Defaults to 'logs/'."
_ARG_NAME_HELP = (
    "Optional name of the log file. If not provided, a name will be generated."
)


def standard_transform() -> None:
    """
    Perform data transformation operations on single-cell data.

    This function orchestrates the data transformation process, including:
    - CPM normalization
    - Log transformation
    - Min-max scaling

    The input data file path, log file directory, and optional log file name can be provided
    through command-line arguments.

    Example:
        Run the following command in the terminal:
        $ python transform.py -d data/input_data.h5ad -l /path/to/logs/ -n log_file_name
    """
    # Add cmd parser
    parser = argparse.ArgumentParser(description=_PARSER_DESC)

    # Add arguments
    parser.add_argument("-d", "--data", type=str, help=_ARG_DATA_HELP)
    parser.add_argument("-l", "--log", type=str, default="logs/", help=_ARG_LOG_HELP)
    parser.add_argument("-n", "--name", type=str, default=None, help=_ARG_NAME_HELP)

    # Parse arguments -> Can be called with args.FLAG
    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger("INFO", log_dir=args.log, log_file=args.name)

    # >>> Functionality starts here
    adata = io_utils.load_adata(args.data, logger)

    # CPM normalization
    cpm_normalized = mtd.sparse_cpm(adata.X, logger)

    # Log transformation
    log_transformed = mtd.sparse_log(cpm_normalized, logger)

    # Min-max normalization
    min_max_scaled = mtd.sparse_min_max(log_transformed, logger=logger)

    # Use the base name of the input file with a modified suffix
    filename = os.path.splitext(os.path.basename(args.data))[0] + "_normalized.h5ad"
    io_utils.save_adata_layer(
        adata,
        transformed_data=min_max_scaled,
        layer_name="min_max_scaled",
        filename=filename,
        logger=logger,
    )


if __name__ == "__main__":
    standard_transform()
