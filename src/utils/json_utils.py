"""
Model Config Utilities.

This module provides functions to handle the loading and assembling of neural network
configurations, including layers, activation functions, loss functions, and optimizers. 
It also supports reading model configurations from JSON files.

Functions:
    - load_config_file: Loads a model configuration file and extracts relevant settings.
    - import_model_params: Reads and returns model parameters from a JSON file.
    - assemble_model: Assembles the model architecture based on a configuration dictionary.
    - import_model_architecture: Imports and assembles the forward and backward layers of the model.
    - assemble_layer: Assembles a single layer based on the provided configuration.
    - assemble_layer_w_actv: Assembles a layer with an activation function.
    - assemble_layer_wo_actv: Assembles a layer without an activation function.
    - assemble_layer_linear: Assembles a linear layer.
    - assemble_layer_activation: Assembles an activation layer.
    - assemble_layer_relu: Assembles a ReLU activation layer.
    - assemble_layer_silu: Assembles a SiLU activation layer.
    - import_loss_function: Imports a loss function based on the provided name.
    - import_optimizer: Imports an optimizer based on the provided name and parameters.
"""

# Standard imports
import json
from logging import Logger
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Third-party imports
import torch.nn as nn
import torch.optim as optim

# TODO: Create validation for JSON schema
# from jsonschema import validate


__author__ = "Christian Kolland"
__version__ = "1.0"

# Constants
_ACTIVATION_FUNCTIONS = ["relu", "silu"]
_LAYER_TYPES = ["linear"]
_LAYER_ROLES = ["input", "hidden", "latent_space", "output"]
_LOSS_FUNCTIONS = ["mse", "bce"]
_OPTIMIZERS = ["adam"]


def load_config_file(
    path_to_conf: str, logger: Optional[Logger] = None
) -> Tuple[str, List[nn.Module], List[nn.Module], nn.Module, Any, int, int]:
    """
    Loads the model configuration from a given file path.

    This function reads a configuration file (expected to be in JSON format) and extracts
    the model-related and training-related settings. If a logger is provided, it logs the
    configuration details.

    Args:
        path_to_conf (str): The path to the configuration file.
        logger (Optional[Logger]): Logger instance for logging information. Defaults to None.

    Returns:
        dict: A dictionary containing the model version, encoder layers, decoder layers,
              loss function, optimizer, batch size, and number of training epochs.
    """
    config = import_model_params(path_to_conf)

    if logger is not None:
        logger.debug(config)
        logger.info("Config file loaded.")

    # Separate model and training configuration
    model_config = config["model"]
    train_config = config["training"]

    model_version = model_config["version"]
    encoder_layers, decoder_layers = assemble_model(model_config["layers"])
    loss_function = import_loss_function(model_config["loss_function"])
    optimizer = model_config["optimization"]
    batch_size = train_config["batch_size"]
    num_epochs = train_config["training_epochs"]

    return (
        model_version,
        encoder_layers,
        decoder_layers,
        loss_function,
        optimizer,
        batch_size,
        num_epochs,
    )


def import_model_params(file_path: str) -> Dict[str, Any]:
    """
    Imports model parameters from a JSON file.

    This function reads a JSON file containing the model parameters and returns them
    as a dictionary.

    Args:
        file_path (str): The path to the JSON file containing the model parameters.

    Returns:
        Dict[str, Any]: A dictionary representation of the JSON file containing the model parameters.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def assemble_model(model_layers: dict) -> Tuple[List[nn.Module], List[nn.Module]]:
    """
    Assembles the model architecture based on the provided configuration.

    This function assembles the encoder and decoder layers of the model based on the
    configuration provided in `model_layers`.

    Args:
        model_layers (dict): Dictionary containing the encoder and decoder layer configurations.

    Returns:
        Tuple[List[nn.Module], List[nn.Module]]: A tuple containing the assembled encoder
                                                 layers and decoder layers.
    """
    encoder_layers, decoder_layers = import_model_architecture(
        forward=model_layers["encoder"],
        backward=model_layers["decoder"],
    )

    return encoder_layers, decoder_layers


def import_model_architecture(
    forward: List[Dict[str, Any]], backward: Optional[List[Dict[str, Any]]] = None
) -> Union[List[nn.Module], Tuple[List[nn.Module], List[nn.Module]]]:
    """
    Import and assemble the model architecture.

    This function assembles the forward layers and optionally the backward layers
    of the model based on the provided configuration.

    Args:
        forward (List[Dict[str, Any]]): List of dictionaries representing the forward layers.
        backward (Optional[List[Dict[str, Any]]], optional): List of dictionaries representing
            the backward layers. Defaults to None.

    Returns:
        Union[List[nn.Module], Tuple[List[nn.Module], List[nn.Module]]]: The assembled forward layers, and optionally the backward layers if provided.
    """
    forward_layers = []
    for layer in forward:
        forward_layers += assemble_layer(layer)

    if backward is None:
        return forward_layers
    else:
        backward_layers = []
        for layer in backward:
            backward_layers += assemble_layer(layer)

        return forward_layers, backward_layers


def assemble_layer(layer: Dict[str, Any]) -> List[nn.Module]:
    """
    Assemble a layer based on the provided configuration.

    This function assembles a neural network layer based on the configuration provided
    in the `layer` dictionary.

    Args:
        layer (Dict[str, Any]): Dictionary containing the configuration for the layer.

    Returns:
        List[nn.Module]: A list containing the assembled layers.

    Raises:
        ValueError: If the layer role is not one of the allowed values.
    """
    # Ensure the value is within the allowed range.
    if layer["role"] not in _LAYER_ROLES:
        raise ValueError(f"Invalid layer role.\nRole must be one of: {_LAYER_ROLES}")

    if layer["role"] in ["input", "hidden"]:
        return assemble_layer_w_actv(
            layer["type"],
            layer["in_dimension"],
            layer["out_dimension"],
            layer["activation"],
        )
    elif layer["role"] in ["latent_space", "output"]:
        return assemble_layer_wo_actv(
            layer["type"],
            layer["in_dimension"],
            layer["out_dimension"],
        )


def assemble_layer_w_actv(
    layer_type: Literal["linear"],
    in_dim: int,
    out_dim: int,
    activation: Literal["relu", "silu"],
) -> List[nn.Module]:
    """
    Assemble a layer with an activation function.

    This function assembles a neural network layer with an activation function
    based on the provided configuration.

    Args:
        layer_type (Literal["linear"]): The type of the layer.
        in_dim (int): The input dimension of the layer.
        out_dim (int): The output dimension of the layer.
        activation (Literal["relu", "silu"]): The activation function to use.

    Returns:
        List[nn.Module]: A list containing the assembled linear layer and the activation layer.

    Raises:
        ValueError: If the layer type is not one of the allowed values.
    """
    # Ensure the value is within the allowed range.
    if layer_type not in _LAYER_TYPES:
        raise ValueError(
            f"Invalid layer type.\nLayer type must be one of: {_LAYER_TYPES}"
        )

    if layer_type == "linear":
        # Return as list for later conversion.
        return [
            assemble_layer_linear(in_dim=in_dim, out_dim=out_dim),
            assemble_layer_activation(activation=activation),
        ]


def assemble_layer_wo_actv(
    layer_type: Literal["linear"], in_dim: int, out_dim: int
) -> List[nn.Module]:
    """
    Assemble a layer without an activation function.

    This function assembles a neural network layer without an activation function
    based on the provided configuration.

    Args:
        layer_type (Literal["linear"]): The type of the layer.
        in_dim (int): The input dimension of the layer.
        out_dim (int): The output dimension of the layer.

    Returns:
        List[nn.Module]: A list containing the assembled linear layer.

    Raises:
        ValueError: If the layer type is not one of the allowed values.
    """
    # Ensure the value is within the allowed range.
    if layer_type not in _LAYER_TYPES:
        raise ValueError(
            f"Invalid layer type.\nLayer type must be one of: {_LAYER_TYPES}"
        )

    if layer_type == "linear":
        # Return as list for later conversion.
        return [assemble_layer_linear(in_dim=in_dim, out_dim=out_dim)]


def assemble_layer_linear(in_dim: int, out_dim: int) -> nn.Module:
    """
    Assembles a linear layer.

    Args:
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.

    Returns:
        nn.Module: The assembled linear layer.
    """
    return nn.Linear(in_dim, out_dim)


def assemble_layer_activation(activation: Literal["relu", "silu"]) -> nn.Module:
    """
    Assembles an activation layer based on the provided activation type.

    Args:
        activation (Literal["relu", "silu"]): The type of activation layer to assemble.

    Returns:
        nn.Module: The assembled activation layer.

    Raises:
        ValueError: If the activation function is not one of the allowed values.
    """
    # Ensure the value is within the allowed range.
    if activation not in _ACTIVATION_FUNCTIONS:
        raise ValueError(
            f"Invalid activation function.\nActivation must be one of: {_ACTIVATION_FUNCTIONS}"
        )

    if activation == "relu":
        return assemble_layer_relu()
    elif activation == "silu":
        return assemble_layer_silu()


def assemble_layer_relu() -> nn.Module:
    """
    Assembles a ReLU activation layer.

    Returns:
        nn.Module: A ReLU activation layer.
    """
    return nn.ReLU()


def assemble_layer_silu() -> nn.Module:
    """
    Assembles a SiLU activation layer.

    Returns:
        nn.Module: A SiLU activation layer.
    """
    return nn.SiLU()


def import_loss_function(loss_function: Literal["mse", "bce"]) -> nn.Module:
    """
    Imports the loss function based on the provided name.

    Args:
        loss_function (Literal["mse", "bce"]): The name of the loss function.

    Returns:
        nn.Module: The corresponding loss function.

    Raises:
        ValueError: If the loss function is not one of the allowed values.
    """
    if loss_function not in _LOSS_FUNCTIONS:
        raise ValueError(
            f"Invalid loss function.\nLoss must be one of: {_LOSS_FUNCTIONS}"
        )

    if loss_function == "mse":
        return nn.MSELoss()
    elif loss_function == "bce":
        return nn.BCELoss()


def import_optimizer(
    params: Any,
    optimizer: Literal["adam"],
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-3,
) -> optim.Optimizer:
    """
    Imports the optimizer based on the provided name and parameters.

    Args:
        params (Any): Parameters to optimize.
        optimizer (str): The name of the optimizer.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-2.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 1e-3.

    Returns:
        optim.Optimizer: The corresponding optimizer.

    Raises:
        ValueError: If the optimizer is not one of the allowed values.
    """
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"Invalid optimizer.\nOptimizer must be one of: {_OPTIMIZERS}")

    if optimizer == "adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
