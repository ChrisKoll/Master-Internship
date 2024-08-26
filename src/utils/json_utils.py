"""
JSON Validation and Model Configuration Module.

This module handles the configuration of neural network layers, optimizers, and 
training parameters using the Pydantic library for data validation. It supports 
the assembly of encoder and decoder layers from configuration files, and includes 
utilities for importing and validating configuration data.

Classes:
    - LayerConfig: Configuration class for individual layers, including type, dimensions, and activation function.
    - CoderConfig: Configuration class for encoder and decoder layers.
    - OptimizerConfig: Configuration class for optimizer settings.
    - ModelConfig: Configuration class for the model, including layers and optimization settings.
    - TrainingConfig: Configuration class for training parameters such as batch size and epochs.
    - Config: Master configuration class that encapsulates the model and training configurations.

Functions:
    - import_config_json: Imports and parses a JSON configuration file into a dictionary.
    - assemble_layers: Assembles layers based on the provided list of LayerConfig objects.
    - assemble_layer: Assembles a single layer based on the LayerConfig object.
"""

# Standard imports
import json
from typing import Tuple

# Third-party imports
from pydantic import BaseModel, field_validator
import torch.nn as nn
import torch.optim as optim


__author__ = "Christian Kolland"
__version__ = "1.2"

# Constants
_LAYER_ROLES = ["input", "hidden", "latent_space", "output"]
_LAYER_TYPES = {"linear": nn.Linear}
_ACTIVATION_FUNCTIONS = {"": None, "relu": nn.ReLU, "silu": nn.SiLU}
_OPTIMIZERS = {"adam": optim.Adam}
_LOSS_FUNCTIONS = {"mse": nn.MSELoss, "bce": nn.BCELoss}


class LayerConfig(BaseModel):
    """
    Configuration for a single neural network layer.

    Attributes:
        role (str): Role of the layer in the network (e.g., input, hidden, latent_space, output).
        type (str): Type of the layer (e.g., linear).
        in_dimension (int): Input dimension of the layer.
        out_dimension (int): Output dimension of the layer.
        activation (str): Activation function applied after the layer (e.g., relu, silu).
    """

    role: str
    type: str
    in_dimension: int
    out_dimension: int
    activation: str

    @field_validator("role")
    def validate_role(cls, value):
        """
        Validates the role of the layer.

        Args:
            value (str): The role of the layer to be validated.

        Returns:
            str: The validated role.

        Raises:
            ValueError: If the role is not one of the allowed values.
        """
        if value in _LAYER_ROLES:
            return value
        else:
            raise ValueError(
                f"Invalid layer role: {value}. Must be one of: {_LAYER_ROLES}"
            )

    @field_validator("type")
    def validate_type(cls, value):
        """
        Validates the type of the layer.

        Args:
            value (str): The type of the layer to be validated.

        Returns:
            type: The corresponding PyTorch layer type.

        Raises:
            ValueError: If the type is not one of the allowed values.
        """
        if value in _LAYER_TYPES.keys():
            return _LAYER_TYPES[value]
        else:
            raise ValueError(
                f"Invalid layer role: {value}. Must be one of: {_LAYER_TYPES.keys()}"
            )

    @field_validator("activation")
    def validate_activation(cls, value):
        """
        Validates the activation function for the layer.

        Args:
            value (str): The activation function to be validated.

        Returns:
            Optional[torch.nn.Module]: The corresponding PyTorch activation function, or None.

        Raises:
            ValueError: If the activation function is not one of the allowed values.
        """
        if value in _ACTIVATION_FUNCTIONS.keys():
            return _ACTIVATION_FUNCTIONS[value]
        else:
            raise ValueError(
                f"Invalid layer role: {value}. Must be one of: {_ACTIVATION_FUNCTIONS.keys()}"
            )


class CoderConfig(BaseModel):
    """
    Configuration for encoder and decoder layers.

    Attributes:
        encoder (list[LayerConfig]): List of layer configurations for the encoder.
        decoder (list[LayerConfig]): List of layer configurations for the decoder.
    """

    encoder: list[LayerConfig]
    decoder: list[LayerConfig]


class OptimizerConfig(BaseModel):
    """
    Configuration for the optimizer used in model training.

    Attributes:
        optimizer (str): Name of the optimizer (e.g., adam).
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
    """

    optimizer: str
    learning_rate: float
    weight_decay: float

    @field_validator("optimizer")
    def validate_optimizer(cls, value):
        """
        Validates the optimizer type.

        Args:
            value (str): The optimizer to be validated.

        Returns:
            type: The corresponding PyTorch optimizer.

        Raises:
            ValueError: If the optimizer is not one of the allowed values.
        """
        if value in _OPTIMIZERS.keys():
            return _OPTIMIZERS[value]
        else:
            raise ValueError(
                f"Invalid optimizer: {value}. Must be one of: {_OPTIMIZERS.keys()}"
            )


class ModelConfig(BaseModel):
    """
    Configuration for the model architecture and training.

    Attributes:
        name (str): Name of the model.
        type (str): Type of the model.
        loss_function (str): Loss function to be used during training.
        layers (CoderConfig): Configuration for the model's layers.
        optimization (OptimizerConfig): Configuration for the model's optimizer.
    """

    name: str
    type: str
    loss_function: str
    layers: CoderConfig
    optimization: OptimizerConfig

    @field_validator("loss_function")
    def validate_loss_function(cls, value):
        """
        Validates the loss function type.

        Args:
            value (str): The loss function to be validated.

        Returns:
            type: The corresponding PyTorch loss function.

        Raises:
            ValueError: If the loss function is not one of the allowed values.
        """
        if value in _LOSS_FUNCTIONS.keys():
            return _LOSS_FUNCTIONS[value]
        else:
            raise ValueError(
                f"Invalid loss function: {value}. Must be one of: {_LOSS_FUNCTIONS.keys()}"
            )


class TrainingConfig(BaseModel):
    """
    Configuration for training parameters.

    Attributes:
        batch_size (int): Number of samples per batch.
        training_epochs (int): Number of epochs for training.
    """

    batch_size: int
    training_epochs: int


class Config(BaseModel):
    """
    Master configuration class encapsulating model and training configurations.

    Attributes:
        model (ModelConfig): Model configuration including layers, loss function, and optimizer.
        training (TrainingConfig): Training configuration including batch size and epochs.
    """

    model: ModelConfig
    training: TrainingConfig


def import_config_json(file_path: str) -> dict:
    """
    Imports and parses a JSON configuration file into a dictionary.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        dict: The configuration data parsed from the JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def assemble_layers(
    layer_configs: list[LayerConfig],
) -> Tuple[list[nn.Module], list[nn.Module]]:
    """
    Assembles layers based on the provided list of LayerConfig objects.

    Args:
        layer_configs (list[LayerConfig]): List of LayerConfig objects for assembling layers.

    Returns:
        Tuple[list[nn.Module], list[nn.Module]]: A tuple containing the assembled layers.
    """
    # Outer loop `for config in layer_configs`
    # Inner loop `for layer in assemble_layer(config)`
    return [layer for config in layer_configs for layer in assemble_layer(config)]


def assemble_layer(layer_config: LayerConfig) -> list[nn.Module]:
    """
    Assembles a single layer based on the LayerConfig object.

    Args:
        layer_config (LayerConfig): Configuration for the layer to be assembled.

    Returns:
        list[nn.Module]: A list containing the layer and its activation function, if applicable.
    """
    layer = [layer_config.type(layer_config.in_dimension, layer_config.out_dimension)]

    if layer_config.activation is not None:
        layer.append(layer_config.activation())

    return layer
