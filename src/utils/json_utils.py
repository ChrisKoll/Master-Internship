"""
This module contains utility functions for loading and assembling neural network models from a configuration file.

The main functionalities provided by this module include:
- Loading a model configuration from a JSON file.
- Assembling a neural network model based on the configuration.
- Importing specific layers, activation functions, loss functions, and optimizers based on the configuration.

Constants:
    - _ACTIVATION_FUNCTIONS: Mapping of supported activation functions to their corresponding PyTorch classes.
    - _LAYER_ROLES: List of supported layer roles in the model.
    - _LAYER_TYPES: Mapping of supported layer types to their corresponding PyTorch classes.
    - _LOSS_FUNCTIONS: Mapping of supported loss functions to their corresponding PyTorch classes.
    - _OPTIMIZERS: Mapping of supported optimizers to their corresponding PyTorch classes.

Functions:
    - log_message: Logs a message using the provided logger.
    - load_config_file: Loads the model configuration from a JSON file.
    - import_model_params: Imports model parameters from a JSON file.
    - assemble_model: Assembles the encoder and decoder layers of the model.
    - assemble_structure: Assembles a list of layers based on the provided configuration.
    - assemble_layer: Assembles a single layer based on the provided configuration.
    - assemble_layer_with_optional_activation: Assembles a layer with an optional activation function.
    - import_loss_function: Imports a loss function based on the provided string identifier.
    - import_optimizer: Imports an optimizer based on the provided string identifier.
"""

# Standard imports
from dataclasses import dataclass
import json
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import torch.nn as nn
import torch.optim as optim

# Self-built modules
from utils.dataclasses import OptimizerConfig

# TODO: Create validation for JSON schema
# from jsonschema import validate


__author__ = "Christian Kolland"
__version__ = "1.1"

# Constants
_ACTIVATION_FUNCTIONS = {"relu": nn.ReLU, "silu": nn.SiLU}
_LAYER_ROLES = ["input", "hidden", "latent_space", "output"]
_LAYER_TYPES = {"linear": nn.Linear}
_LOSS_FUNCTIONS = {"mse": nn.MSELoss, "bce": nn.BCELoss}
_OPTIMIZERS = {"adam": optim.Adam}


@dataclass
class ModelConfig:
    """
    Data class representing the model configuration.

    Attributes:
        version (str): Version of the model.
        encoder_layers (List[nn.Module]): List of encoder layers.
        decoder_layers (List[nn.Module]): List of decoder layers.
        loss_function (nn.Module): Loss function used in the model.
        optimizer (Any): Optimizer used for training the model.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
    """

    version: str
    encoder_layers: List[nn.Module]
    decoder_layers: List[nn.Module]
    loss_function: nn.Module
    optimizer: Any
    batch_size: int
    num_epochs: int


def log_message(logger: Optional[Logger], message: str, level: str = "info") -> None:
    """
    Log a message using the provided logger.

    Args:
        logger (Optional[Logger]): Logger instance for logging information. If None, no logging is performed.
        message (str): The message to log.
        level (str, optional): The logging level. Defaults to "info".

    Returns:
        None
    """
    if logger:
        getattr(logger, level)(message)


def load_config_file(path_to_conf: str, logger: Optional[Logger] = None) -> ModelConfig:
    """
    Load the model configuration from a JSON file.

    This function loads a configuration file, extracts the model-related and
    training-related configurations, and returns them as a ModelConfig instance.

    Args:
        path_to_conf (str): Path to the configuration file.
        logger (Optional[Logger]): Logger instance for logging information. Defaults to None.

    Returns:
        ModelConfig: An instance of ModelConfig containing the model and training configurations.

    Raises:
        ValueError: If any configuration is invalid.
    """
    config = import_model_params(path_to_conf)
    log_message(logger, config, "debug")
    log_message(logger, "Config file loaded.", "info")

    model_config = config["model"]
    train_config = config["training"]

    encoder_layers, decoder_layers = assemble_model(model_config["layers"])

    return ModelConfig(
        version=model_config["version"],
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        loss_function=import_loss_function(model_config["loss_function"]),
        optimizer=import_optimizer(model_config["optimization"]),
        batch_size=train_config["batch_size"],
        num_epochs=train_config["training_epochs"],
    )


def import_model_params(file_path: str) -> Dict[str, Any]:
    """
    Assemble the encoder and decoder layers of the model based on the configuration.

    Args:
        layers_config (dict): Dictionary containing the encoder and decoder layer configurations.

    Returns:
        Tuple[List[nn.Module], List[nn.Module]]: A tuple containing the assembled encoder and decoder layers.

    """
    with open(file_path, "r") as file:
        return json.load(file)


def assemble_model(layers_config: dict) -> Tuple[List[nn.Module], List[nn.Module]]:
    """
    Assemble the encoder and decoder layers of the model based on the configuration.

    Args:
        layers_config (dict): Dictionary containing the encoder and decoder layer configurations.

    Returns:
        Tuple[List[nn.Module], List[nn.Module]]: A tuple containing the assembled encoder and decoder layers.
    """
    return (
        assemble_structure(layers_config["encoder"]),
        assemble_structure(layers_config["decoder"]),
    )


def assemble_structure(layers: List[Dict[str, Any]]) -> List[nn.Module]:
    """
    Assemble a list of layers based on the provided configuration.

    Args:
        layers (List[Dict[str, Any]]): List of dictionaries representing the layer configurations.

    Returns:
        List[nn.Module]: A list of assembled layers.
    """
    return [layer for config in layers for layer in assemble_layer(config)]


def assemble_layer(layer: Dict[str, Any]) -> List[nn.Module]:
    """
    Assemble a single layer based on the provided configuration.

    This function handles both the layer type and the optional activation function.

    Args:
        layer (Dict[str, Any]): Dictionary containing the configuration for the layer.

    Returns:
        List[nn.Module]: A list containing the assembled layer(s).

    Raises:
        ValueError: If the layer role or type is not valid.
    """
    if layer["role"] not in _LAYER_ROLES:
        raise ValueError(
            f"Invalid layer role: {layer['role']}. Must be one of: {_LAYER_ROLES}"
        )

    return assemble_layer_with_optional_activation(
        layer["type"],
        layer["in_dimension"],
        layer["out_dimension"],
        layer.get("activation"),
    )


def assemble_layer_with_optional_activation(
    layer_type: str, in_dim: int, out_dim: int, activation: Optional[str] = None
) -> List[nn.Module]:
    """
    Assemble a layer with an optional activation function.

    Args:
        layer_type (str): The type of the layer (e.g., 'linear').
        in_dim (int): The input dimension of the layer.
        out_dim (int): The output dimension of the layer.
        activation (Optional[str], optional): The activation function to use. Defaults to None.

    Returns:
        List[nn.Module]: A list containing the assembled layer and, if specified, the activation layer.

    Raises:
        ValueError: If the layer type or activation function is not valid.
    """
    if layer_type not in _LAYER_TYPES:
        raise ValueError(
            f"Invalid layer type: {layer_type}. Must be one of: {list(_LAYER_TYPES.keys())}"
        )

    layers = [_LAYER_TYPES[layer_type](in_dim, out_dim)]

    if activation:
        if activation not in _ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Invalid activation function: {activation}. Must be one of: {list(_ACTIVATION_FUNCTIONS.keys())}"
            )
        layers.append(_ACTIVATION_FUNCTIONS[activation]())

    return layers


def import_loss_function(loss_function: str) -> nn.Module:
    """
    Import a loss function based on the provided string identifier.

    Args:
        loss_function (str): The identifier for the loss function (e.g., 'mse', 'bce').

    Returns:
        nn.Module: The PyTorch loss function module corresponding to the identifier.

    Raises:
        ValueError: If the loss function identifier is not valid.
    """
    if loss_function not in _LOSS_FUNCTIONS:
        raise ValueError(
            f"Invalid loss function: {loss_function}. Must be one of: {list(_LOSS_FUNCTIONS.keys())}"
        )
    return _LOSS_FUNCTIONS[loss_function]()


def import_optimizer(parameters):
    """Docstring."""
    return OptimizerConfig(
        parameters["optimizer"], parameters["learning_rate"], parameters["weight_decay"]
    )


def configure_optimizer(
    params: Any, optimizer: str, learning_rate: float = 1e-2, weight_decay: float = 1e-3
) -> optim.Optimizer:
    """
    Import an optimizer based on the provided string identifier.

    Args:
        params (Any): The parameters to be optimized.
        optimizer (str): The identifier for the optimizer (e.g., 'adam').
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-2.
        weight_decay (float, optional): The weight decay (L2 penalty) for the optimizer. Defaults to 1e-3.

    Returns:
        optim.Optimizer: The PyTorch optimizer corresponding to the identifier.

    Raises:
        ValueError: If the optimizer identifier is not valid.
    """
    if optimizer not in _OPTIMIZERS:
        raise ValueError(
            f"Invalid optimizer: {optimizer}. Must be one of: {list(_OPTIMIZERS.keys())}"
        )
    return _OPTIMIZERS[optimizer](params, lr=learning_rate, weight_decay=weight_decay)
