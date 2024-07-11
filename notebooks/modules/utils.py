""" This module contains helper functions used in various parts of the project.

Typical usage example:

    dict = import_model(path_to_json)
"""

## Standard
import json
from typing import Literal

# from jsonschema import validate
import torch.nn as nn
import torch.optim as optim

__author__ = "Christian Kolland"
__version__ = "0.0.1"


def import_model_params(file_path: str) -> dict:
    """Imports the Pytorch model information from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A dict representation of the JSON file.

        Insert example!
    """
    with open(file_path, "r") as file:
        return json.load(file)


def assess_layer(layer: dict):
    """Assess the layer structure."""
    allowed_roles = ["input", "hidden", "output", "latent_space"]
    if layer["role"] not in allowed_roles:
        raise ValueError(f"Layer role should be one of {allowed_roles}")

    allowed_types = ["linear"]
    if layer["type"] not in allowed_types:
        raise ValueError(f"Layer type should be one of {allowed_types}")

    allowed_activations = ["silu"]
    if "activation" in layer.keys() and layer["activation"] not in allowed_activations:
        raise ValueError(f"Layer type should be one of {allowed_activations}")

    if layer["role"] == "input":
        if layer["type"] == "linear" and layer["activation"] == "silu":
            return [
                nn.Linear(layer["in_dimension"], layer["out_dimension"]),
                nn.SiLU(),
            ]
    elif layer["role"] == "hidden":
        if layer["type"] == "linear" and layer["activation"] == "silu":
            return [
                nn.Linear(layer["in_dimension"], layer["out_dimension"]),
                nn.SiLU(),
            ]
    elif layer["role"] == "latent_space":
        if layer["type"] == "linear":
            return [nn.Linear(layer["in_dimension"], layer["out_dimension"])]
    elif layer["role"] == "output":
        if layer["type"] == "linear":
            return [nn.Linear(layer["in_dimension"], layer["out_dimension"])]


def import_model_architecture(forward: dict, backward=None):
    """Imports the model archtiecture."""
    forward_layers = []
    for layer in forward:
        forward_layers += assess_layer(layer)

    if backward is None:
        return forward_layers
    else:
        backward_layers = []
        for layer in backward:
            backward_layers += assess_layer(layer)

        return forward_layers, backward_layers


def import_loss_function(loss_function: Literal["mse", "bce"]) -> nn.Module:
    """Import the loss function from JSON structure."""
    allowed_values = ["mse", "bce"]
    if loss_function not in allowed_values:
        raise ValueError(f"loss function should be one of {allowed_values}")

    if loss_function == "mse":
        return nn.MSELoss()
    elif loss_function == "bce":
        return nn.BCELoss()


def import_optimizer(
    params, optimizer: str, learning_rate: float = 1e-2, weight_decay: float = 1e-3
) -> optim.Optimizer:
    """Imports the optimizer from the JSON file."""
    allowed_values = ["adam"]
    if optimizer not in allowed_values:
        raise ValueError(f"optimizer should be one of {allowed_values}")

    if optimizer == "adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
