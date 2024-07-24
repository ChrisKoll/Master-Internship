"""
Module for assembling and importing PyTorch model parameters and components.

This module provides functions to import model parameters from JSON files,
assemble various types of neural network layers (with and without activation functions),
import loss functions, and create optimizers. The functions are designed to work
together to facilitate the construction and configuration of neural network models
in PyTorch.

Functions:
    import_model_params(file_path: str) -> Dict[str, Any]:
        Imports the PyTorch model information from a JSON file.

    assemble_layer_relu() -> nn.Module:
        Assembles a ReLU activation layer.

    assemble_layer_silu() -> nn.Module:
        Assembles a SiLU activation layer.

    assemble_layer_activation(activation: Literal["relu", "silu"]) -> nn.Module:
        Assembles an activation layer based on the provided activation type.

    assemble_layer_linear(in_dim: int, out_dim: int) -> nn.Module:
        Assembles a linear layer.

    assemble_layer_w_actv(
        layer_type: Literal["linear"], in_dim: int, out_dim: int, activation: Literal["relu", "silu"]
    ) -> List[nn.Module]:
        Assembles a layer with an activation function.

    assemble_layer_wo_actv(layer_type: Literal["linear"], in_dim: int, out_dim: int) -> List[nn.Module]:
        Assembles a layer without an activation function.

    assemble_layer(layer: Dict[str, Any]) -> List[nn.Module]:
        Assembles a layer based on the provided configuration.

    import_model_architecture(
        forward: List[Dict[str, Any]], backward: Optional[List[Dict[str, Any]]] = None
    ) -> Union[List[nn.Module], Tuple[List[nn.Module], List[nn.Module]]]:
        Imports the model architecture.

    import_loss_function(loss_function: Literal["mse", "bce"]) -> nn.Module:
        Imports the loss function based on the provided name.

    import_optimizer(
        params: Any, optimizer: str, learning_rate: float = 1e-2, weight_decay: float = 1e-3
    ) -> optim.Optimizer:
        Imports the optimizer based on the provided name and parameters.

Examples:
    Importing model parameters from a JSON file:
        >>> params = import_model_params('path/to/model.json')
        >>> print(params)

    Assembling a linear layer with ReLU activation:
        >>> layer = assemble_layer_w_actv('linear', 128, 64, 'relu')
        >>> print(layer)

    Importing a mean squared error loss function:
        >>> loss_fn = import_loss_function('mse')
        >>> print(loss_fn)

    Creating an Adam optimizer:
        >>> model_params = [param for param in model.parameters()]
        >>> optimizer = import_optimizer(model_params, 'adam')
        >>> print(optimizer)

Note:
    This module requires PyTorch to be installed.
"""

## Standard imports
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

## Third-party imports
import torch.nn as nn
import torch.optim as optim

## TODO: Create validation for JSON schema
# from jsonschema import validate


__author__ = "Christian Kolland"
__version__ = "0.0.2"

## Constants
_ACTIVATION_FUNCTIONS = ["relu", "silu"]
_LAYER_TYPES = ["linear"]
_LAYER_ROLES = ["input", "hidden", "latent_space", "output"]
_LOSS_FUNCTIONS = ["mse", "bce"]
_OPTIMIZERS = ["adam"]


def import_model_params(file_path: str) -> Dict[str, Any]:
    """Imports the PyTorch model information from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: A dictionary representation of the JSON file.

    Example:
        >>> params = import_model_params('path/to/model.json')
        >>> print(params)
        {'layer1': {'role': 'input', 'type': 'linear', 'in_dimension': 128, 'out_dimension': 64, 'activation': 'relu'}}
    """
    with open(file_path, "r") as file:
        return json.load(file)


def assemble_layer_relu() -> nn.Module:
    """Assembles a ReLU activation layer.

    Returns:
        nn.Module: A ReLU activation layer.
    """
    return nn.ReLU()


def assemble_layer_silu() -> nn.Module:
    """Assembles a SiLU activation layer.

    Returns:
        nn.Module: A SiLU activation layer.
    """
    return nn.SiLU()


def assemble_layer_activation(activation: Literal["relu", "silu"]) -> nn.Module:
    """Assembles an activation layer based on the provided activation type.

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


def assemble_layer_linear(in_dim: int, out_dim: int) -> nn.Module:
    """Assembles a linear layer.

    Args:
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.

    Returns:
        nn.Module: The assembled linear layer.
    """
    return nn.Linear(in_dim, out_dim)


def assemble_layer_w_actv(
    layer_type: Literal["linear"],
    in_dim: int,
    out_dim: int,
    activation: Literal["relu", "silu"],
) -> List[nn.Module]:
    """Assembles a layer with an activation function.

    Args:
        layer_type (Literal["linear"]): The type of the layer.
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.
        activation (Literal["relu", "silu"]): The activation function to use.

    Returns:
        List[nn.Module]: A list containing the neural layer and the activation layer.

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
    """Assembles a layer without an activation function.

    Args:
        layer_type (Literal["linear"]): The type of the layer.
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.

    Returns:
        List[nn.Module]: A list containing the neural layer.

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


def assemble_layer(layer: Dict[str, Any]) -> List[nn.Module]:
    """Assembles a layer based on the provided configuration.

    Args:
        layer (Dict[str, Any]): The configuration dictionary for the layer.

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


def import_model_architecture(
    forward: List[Dict[str, Any]], backward: Optional[List[Dict[str, Any]]] = None
) -> Union[List[nn.Module], Tuple[List[nn.Module], List[nn.Module]]]:
    """Imports the model architecture.

    Args:
        forward (List[Dict[str, Any]]): List of dictionaries representing the forward layers.
        backward (Optional[List[Dict[str, Any]]], optional): List of dictionaries representing the backward layers. Defaults to None.

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


def import_loss_function(loss_function: Literal["mse", "bce"]) -> nn.Module:
    """Imports the loss function based on the provided name.

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
    """Imports the optimizer based on the provided name and parameters.

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
