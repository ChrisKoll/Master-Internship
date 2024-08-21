"""
Module Name: logger_setup

This module provides functions for setting up loggers with different logging levels
(INFO and DEBUG) that can log messages to both console and files. The setup functions 
ensure that logs are formatted consistently and that the log files are saved in a 
specified directory.

Functions:
    - setup_logger: Configures a logger for general information logging.
    - setup_debug_logger: Configures a logger for debugging purposes with more detailed output.
"""

# Standard imports
import datetime as dt
import logging
import os
from typing import Optional

__author__ = "Christian Kolland"
__version__ = 1.0


def setup_logger(log_dir: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures a logger to log messages with an INFO level to both the console and a file.

    This function sets up a logger named 'run_logger' that logs messages with an INFO
    level. It ensures that the log directory exists and creates it if necessary. Logs
    are saved in the specified log file or, if no file is provided, a file with a
    timestamped name is created.

    Args:
        log_dir (str): The directory where the log file will be saved.
        log_file (Optional[str]): The name of the log file. If not provided, a file
            with a timestamped name will be created.

    Returns:
        logging.Logger: The configured logger instance.
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


def setup_debug_logger(log_dir: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures a logger to log messages with a DEBUG level to both the console and a file.

    This function sets up a logger named 'debug_logger' that logs messages with a DEBUG
    level. The log directory is checked and created if it doesn't exist. Logs are saved
    in the specified log file or, if no file is provided, a file with a timestamped name
    is created. This logger is ideal for debugging purposes as it captures more detailed
    information.

    Args:
        log_dir (str): The directory where the log file will be saved.
        log_file (Optional[str]): The name of the log file. If not provided, a file with
            a timestamped name will be created.

    Returns:
        logging.Logger: The configured logger instance.
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
    file_handler.setLevel(logging.ERROR)

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
