"""
Autoencoder Training Script.

This script serves as the entry point for training an autoencoder model using single-cell 
data stored in an AnnData object. It allows configuration via command-line arguments and 
JSON configuration files, facilitating the training process, model setup, logging, 
and device management.
The autoencoder model is trained across multiple data folds, each corresponding to a 
donor in the dataset. The script supports logging to a specified file and outputs detailed 
debug and information messages to aid in model development and performance tracking.

Usage:
    - This script is designed to be executed from the command line.
    - The following command-line arguments can be passed:
        -d or --data: Path to the data file containing single-cell data (AnnData format).
        -l or --layer: Name of the specific data layer in AnnData to be used for training.
        -c or --conf: Path to the JSON configuration file that defines model and training
            parameters.
        -x or --log: Directory path where the log file will be stored. Defaults to 'logs/'.
        -f or --name: Optional name for the log file. If not provided, a name will be generated
            automatically.

Functions:
    - main: The main function that parses arguments, initializes the logger, loads data,
                sets up the model, and runs the training process.
"""

# Standard imports
import argparse

# Third-party imports
import torch

# Self-built modules
import src.autoencoder.ae_model as ae
from src.utils.data_utils import TrainingConfig
import src.utils.training_utils as T
import src.utils.io_utils as ioutils
from src.utils.logging_utils import setup_logger
import src.utils.json_utils as jutils


__author__ = "Christian Kolland"
__version__ = "1.0"

# Constants
_PARSER_DESC = "Script to train a Autoencoder (AE) model."
_ARG_DATA_HELP = "Path to data file to be used for training VAE."
_ARG_LAYER_HELP = "Name of data layer that will be used for model fitting."
_ARG_CONF_HELP = "Path to model configuration JSON file."
_ARG_LOG_HELP = "Directory path where log file will be saved. Defaults to 'logs/'."
_ARG_NAME_HELP = "Optional name of log file. If not provided, a name is generated."


def main() -> None:
    """
    Main function to initiate training of the autoencoder model.

    Command-line Arguments:
    -d or --data: Path to the AnnData file.
    -l or --layer: Name of the specific data layer from the AnnData object.
    -c or --conf: Path to the JSON configuration file.
    -x or --log: Directory where logs will be saved (default: 'logs/').
    -f or --name: Optional log file name (auto-generated if not provided).
    """
    # Add cmd parser
    parser = argparse.ArgumentParser(description=_PARSER_DESC)

    # Define and add command-line arguments
    parser.add_argument("-d", "--data", type=str, help=_ARG_DATA_HELP)
    parser.add_argument("-l", "--layer", type=str, help=_ARG_LAYER_HELP)
    parser.add_argument("-c", "--conf", type=str, help=_ARG_CONF_HELP)
    parser.add_argument("-x", "--log", type=str, default="logs/", help=_ARG_LOG_HELP)
    parser.add_argument("-f", "--name", type=str, default=None, help=_ARG_NAME_HELP)

    # Parse arguments -> args.FLAG can be used to access arguments
    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger("INFO", log_dir=args.log, log_file=args.name)

    # >>> Functionality starts here
    # Determine whether to use GPU or CPU for model computation
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the AnnData object from the specified file
    adata = ioutils.load_adata(args.data, logger)

    # Load the JSON configuration file
    config_json = jutils.import_config_json(args.conf)
    config = jutils.Config(**config_json)

    logger.debug(config)
    logger.info("Config loaded")

    # Assemble encoder and decoder layers for the autoencoder model
    encoder_layers = jutils.assemble_layers(config.model.layers.encoder)
    decoder_layers = jutils.assemble_layers(config.model.layers.decoder)
    # Assemble model
    model = ae.Autoencoder(encoder_layers, decoder_layers, config.model.loss_function())

    logger.debug(model)
    logger.info("Model assembled")

    # List of donors to be used for the testing process
    donors = ["D1", "H2", "D5"]

    # Initialize the training configuration with provided parameters
    training_conf = TrainingConfig(
        donors,
        config.training.batch_size,
        config.training.training_epochs,
        device,
    )

    # Setup training instance with the model, data, and configurations
    training = T.Training(
        model,
        config.model.name,
        adata,
        args.layer,
        config.model.optimization,
        training_conf,
        logger,
    )

    # Start training process
    training.fit()
    logger.info("Training process completed successfully!")


if __name__ == "__main__":
    main()
