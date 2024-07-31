"""
Logging configuration module.

This module provides functions to set up and configure loggers for different purposes.
Two types of loggers are provided:

1. `setup_debug_logger`:
   - Configures a logger for debugging purposes.
   - Logs messages at DEBUG level or higher.
   - Outputs to both a log file and the console.
   - If no log file name is provided, a default name with a timestamp is used.

2. `setup_run_logger`:
   - Configures a logger for general runtime logging.
   - Logs messages at INFO level or higher.
   - Outputs to both a log file and the console.
   - If no log file name is provided, a default name with a timestamp is used.

Usage:
    To configure a logger, call the respective function with the desired log directory and optional log file name.

Example:
    >>> from logging_config import setup_debug_logger
    >>> logger = setup_debug_logger('/path/to/log/dir', 'debug.log')
    >>> logger.debug('This is a debug message.')

    >>> from logging_config import setup_run_logger
    >>> logger = setup_run_logger('/path/to/log/dir', 'run.log')
    >>> logger.info('This is an info message.')
"""

# Standard imports
import datetime as dt
import logging
import os
from typing import Optional

__author__ = "Christian Kolland"
__version__ = 1.1


def setup_debug_logger(log_dir: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Sets up a logger for debugging.

    This logger will log messages at DEBUG level or higher to both a log file
    and the console. If `log_file` is not provided, the log file will be named
    with a timestamp indicating the run time.

    Args:
        log_dir (str): Directory where the log file will be stored.
        log_file (Optional[str]): Name of the log file. If not provided, a default name
                                  with the current timestamp will be used.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("debug_logger")
    logger.setLevel(logging.DEBUG)

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
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s :: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_run_logger(log_dir: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Sets up a logger for logging script runs.

    This logger will log messages at INFO level or higher to both a log file
    and the console. If `log_file` is not provided, the log file will be named
    with a timestamp indicating the run time.

    Args:
        log_dir (str): Directory where the log file will be stored.
        log_file (Optional[str]): Name of the log file. If not provided, a default name
                                  with the current timestamp will be used.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger("run_logger")
    logger.setLevel(logging.INFO)  # Not for debugging

    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # If not create

    # Assemble log file path
    if log_file is not None:
        file_path = os.path.join(log_dir, log_file)
    else:
        log_file = f"run_{dt.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.log"
        file_path = os.path.join(log_dir, log_file)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)  # Not for debugging

    # Create console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Not for debugging

    # Define the logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s :: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
