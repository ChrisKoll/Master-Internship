"""Docstring."""

# Standard imports
import argparse
from datetime import datetime
import os

# Third-party imports
import torch

# Self-built modules
import src.autoencoder.ae_model as ae
import src.autoencoder.ae_training as T
import src.utils.io_utils as ioutils
from src.utils.logging_utils import setup_logger
import src.utils.json_utils as jutils


__author__ = "Christian Kolland"
__version__ = "1.0"

# Constants
_PARSER_DESC = "Parser description."
_ARG_DATA_HELP = "Path to the data file."
_ARG_LAYER_HELP = "Name of data layer used for fitting."
_ARG_CONF_HELP = "Path to the configuration file."
_ARG_LOG_HELP = "Directory path where the log file will be saved. Defaults to 'logs/'."
_ARG_NAME_HELP = (
    "Optional name of the log file. If not provided, a name will be generated."
)


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

    # Parse arguments -> Can be called with args.FLAG
    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger("INFO", log_dir=args.log, log_file=args.name)

    # >>> Functionality starts here
    # Established the type of device used for model processing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adata = ioutils.load_adata(args.data, logger)
    config_json = jutils.import_config_json(args.conf)
    config = jutils.Config(**config_json)

    logger.debug(config)
    logger.info("Config loaded")

    encoder_layers = jutils.assemble_layers(config.model.layers.encoder)
    decoder_layers = jutils.assemble_layers(config.model.layers.decoder)
    # Assemble model
    model = ae.Autoencoder(encoder_layers, decoder_layers, config.model.loss_function())
    model.to(device)

    logger.debug(model)
    logger.info("Model assembled")

    T.fit(
        model,
        config.model.name,
        adata,
        args.layer,
        config.model.optimization,
        config.training.batch_size,
        config.training.training_epochs,
        device,
        logger,
    )


if __name__ == "__main__":
    main()
