"""
Logging Utilities Module.

This module provides functions for setting up configurable loggers that can log
messages to both console and files. The setup function ensures that logs are
formatted consistently and that the log files are saved in a specified directory.

Functions:
    - setup_logger: Configures a logger with customizable logging level and file logging option.
    - log_message: Logs a message using the provided logger.
"""

# Standard imports
import datetime as dt
import logging as log
import os
from typing import Optional

__author__ = "Christian Kolland"
__version__ = 1.0


def setup_logger(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs/",
    log_file: Optional[str] = None,
) -> log.Logger:
    """
    Configures a logger with customizable logging level and file logging option.

    This function sets up a logger with the specified logging level. It always logs
    to the console and optionally logs to a file if log_to_file is True. The log
    directory is checked and created if it doesn't exist. If no log file name is
    provided, it creates a file with a timestamped name.

    Args:
        level (str): The logging level to use (e.g., "DEBUG", "INFO"). Defaults to "INFO".
        log_to_file (bool): Whether to log messages to a file. Defaults to True.
        log_dir (str): The directory where the log file will be saved. Defaults to "logs/".
        log_file (Optional[str]): The name of the log file. If not provided, a file with
            a timestamped name will be created.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = log.getLogger(f"custom_{level.lower()}_logger")
    logger.propagate = False  # Prevent propagation to parent loggers
    logger.handlers = []  # Remove any existing handlers - Jupyter Notebooks

    # Evaluate logging level
    level = getattr(log, level)
    logger.setLevel(level)

    # Define logging format
    formatter = log.Formatter("%(asctime)s - %(levelname)s :: %(message)s")

    # Create console handler for logging to the console
    console_handler = log.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        # Ensure the log directory exists
        if not os.path.exists(log_dir):  # Check log dir exists
            os.makedirs(log_dir)  # If not create

        # Assemble log file path
        if log_file is not None:
            file_path = os.path.join(log_dir, log_file)
        else:
            log_file = f"debug_{dt.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.log"
            file_path = os.path.join(log_dir, log_file)

        # Create file handler for logging to a file
        file_handler = log.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_message(
    message: str, logger: Optional[log.Logger] = None, level: str = "info"
) -> None:
    """
    Log a message using the provided logger.

    Args:
        message (str): The message to log.
        logger (Optional[Logger]): Logger instance for logging information.
            If None, no logging is performed.
        level (str): The logging level to use. Defaults to "info".

    Returns:
        None
    """
    if logger:
        getattr(logger, level)(message)


def log_train_metrics(
    writer, fold, total_norm, loss, num_upd, recon_loss=None, kl_loss=None
) -> None:
    """Docstring."""
    writer.add_scalar(f"{fold}/Train/GradNorm", total_norm, num_upd)
    writer.add_scalar(f"{fold}/Train/Loss", loss, num_upd)

    if recon_loss is not None and kl_loss is not None:
        writer.add_scalar(f"{fold}/Train/Loss/Recon", recon_loss, num_upd)
        writer.add_scalar(f"{fold}/Train/Loss/KLD", kl_loss, num_upd)


def log_val_metrics(writer, fold, loss, num_upd, recon_loss=None, kl_loss=None) -> None:
    """Docstring."""
    writer.add_scalar(f"{fold}/Val/Loss", loss, num_upd)

    # For VAE
    if recon_loss is not None and kl_loss is not None:
        writer.add_scalar(f"{fold}/Val/Loss/Recon", recon_loss, num_upd)
        writer.add_scalar(f"{fold}/Val/Loss/KLD", kl_loss, num_upd)


def log_test_metrics(writer, fold, loss, recon_loss=None, kl_loss=None) -> None:
    """Docstring."""
    writer.add_scalar(f"{fold}/Test/Loss", loss)

    # For VAE
    if recon_loss is not None and kl_loss is not None:
        writer.add_scalar(f"{fold}/Test/Loss/Recon", recon_loss)
        writer.add_scalar(f"{fold}/Test/Loss/KLD", kl_loss)
