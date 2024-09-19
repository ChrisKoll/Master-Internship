"""
VAE Training Script.

This script facilitates the training of a Variational Autoencoder (VAE) model 
by loading configuration settings, assembling the model, and managing the training process. 
The script utilizes command-line arguments for specifying data paths, configuration files, 
and logging settings. The training can be customized based on the input configurations.

Usage:
    The script can be executed via the command line with the following arguments:
        -d or --data: Path to the data file (required).
        -l or --layer: Name of the data layer used for fitting (required).
        -c or --conf: Path to the configuration JSON file (required).
        -x or --log: Directory path where the log file will be saved (default: 'logs/').
        -f or --name: Optional name for the log file (default: auto-generated).

Functions:
    - main: The entry point function, responsible for handling argument parsing,
            setting up the environment, loading data and configuration files, 
            and managing the model training process.
"""

# Standard imports
import argparse

# Third-party imports
import torch

# Self-built modules
import src.variational_autoencoder.vae_model as vae
from src.utils.data_utils import TrainingConfig
import src.utils.training_utils as T
import src.utils.io_utils as ioutils
from src.utils.logging_utils import setup_logger
import src.utils.json_utils as jutils


__author__ = "Christian Kolland"
__version__ = "1.0"

# Constants
_PARSER_DESC = "Script to train a Variational Autoencoder (VAE) model."
_ARG_DATA_HELP = "Path to data file to be used for training VAE."
_ARG_LAYER_HELP = "Name of data layer that will be used for model fitting."
_ARG_CONF_HELP = "Path to model configuration JSON file."
_ARG_LOG_HELP = "Directory path where log file will be saved. Defaults to 'logs/'."
_ARG_NAME_HELP = "Optional name of log file. If not provided, a name is generated."


def main() -> None:
    """Docstring."""
    # Add cmd parser
    parser = argparse.ArgumentParser(description=_PARSER_DESC)

    # Add arguments
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
    model = vae.VariationalAutoencoder(
        encoder_layers,
        decoder_layers,
        config.model.loss_function(),
    )

    logger.debug(model)
    logger.info("Model assembled")

    # List of donors to be used for the testing process
    # donors = ["D1", "H2", "D5"]
    donors = ["D1"]

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
