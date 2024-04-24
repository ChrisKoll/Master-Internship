"""
TRAIN module - MODEL:
Contains functionality to generate an autoencoder neural network.
The code can be used to create variable size networks for normal and variational autocoders.
"""

# Standard library imports
from typing import Optional

# Third-party library imports
import torch
import torch.nn as nn
import torch.nn.functional as f


class Encoder(nn.Module):
    """
    Class containing the encoder.
    """

    def __init__(
        self,
        size_input_layer: int,
        size_layer_one: int,
        size_layer_two: int,
        size_layer_three: int,
        size_latent_space: int,
    ):
        """Constructor"""
        super().__init__()

        # Model architecture
        self.input_layer = nn.Linear(size_input_layer, size_layer_one)
        self.hidden_layer1 = nn.Linear(size_layer_one, size_layer_two)
        self.hidden_layer2 = nn.Linear(size_layer_two, size_layer_three)
        # Latent space
        self.latent_space = nn.Linear(size_layer_three, size_latent_space)

    def forward(self, x):
        """
        Forward function for the encoder model.

        :param x: Data tensor
        :return: Returns latent space z
        """
        # Reduce input tensor to 1 dimension
        # x = torch.flatten(x, start_dim=1)
        # ReLU activation
        output = f.relu(self.input_layer(x))
        output = f.relu(self.hidden_layer1(output))
        output = f.relu(self.hidden_layer2(output))

        # Dimension reduced input
        z = self.latent_space(output)

        return z


class VariationalEncoder(nn.Module):
    """
    Class containing the variational encoder.
    """

    def __init__(
        self,
        size_input_layer: int,
        size_layer_one: int,
        size_layer_two: int,
        size_layer_three: int,
        size_latent_space: int,
    ):
        """Constructor"""
        super().__init__()

        # Model architecture
        self.input_layer = nn.Linear(size_input_layer, size_layer_one)
        self.hidden_layer1 = nn.Linear(size_layer_one, size_layer_two)
        self.hidden_layer2 = nn.Linear(size_layer_two, size_layer_three)
        # Latent space for mean and standard deviation
        self.latent_space1 = nn.Linear(size_layer_three, size_latent_space)
        self.latent_space2 = nn.Linear(size_layer_three, size_latent_space)

    def forward(self, x):
        """
        Forward function for the encoder model.

        :param x: Data tensor
        :return: Returns latent space z
        """
        # Reduce input tensor to 1 dimension
        # x = torch.flatten(x, start_dim=1)
        # ReLU activation
        output = f.relu(self.input_layer(x))
        output = f.relu(self.hidden_layer1(output))
        output = f.relu(self.hidden_layer2(output))

        # Mean vector
        mu = self.latent_space1(output)
        # Vector of standard deviation
        # Exponential activation
        sigma = torch.exp(self.latent_space2(output))
        # Combined latent space
        z = torch.distributions.Normal(mu, sigma)

        return z, mu, sigma


class Decoder(nn.Module):
    """
    Class containing the decoder.
    """

    def __init__(
        self,
        size_input_layer: int,
        size_layer_one: int,
        size_layer_two: int,
        size_layer_three: int,
        size_latent_space: int,
    ):
        """Constructor"""
        super().__init__()

        # Model architecture
        self.input_layer = nn.Linear(size_latent_space, size_layer_three)
        self.hidden_layer1 = nn.Linear(size_layer_three, size_layer_two)
        self.hidden_layer2 = nn.Linear(size_layer_two, size_layer_one)
        self.output_layer = nn.Linear(size_layer_one, size_input_layer)

    def forward(self, z):
        """
        Forward function for the decoder model.

        :param z: Latent space representation
        :return: Original representation of the data
        """
        # ReLU activation
        output = f.relu(self.input_layer(z))
        output = f.relu(self.hidden_layer1(output))
        output = f.relu(self.hidden_layer2(output))
        output = self.output_layer(output)

        return output


class Autoencoder(nn.Module):
    """
    Class containing the whole autoencoder.
    """

    def __init__(
        self,
        size_input_layer: int,
        size_layer_one: int,
        size_layer_two: int,
        size_layer_three: int,
        size_latent_space: int,
    ):
        """Constructor"""
        super().__init__()

        self.encoder = Encoder(
            size_input_layer,
            size_layer_one,
            size_layer_two,
            size_layer_three,
            size_latent_space,
        )
        self.decoder = Decoder(
            size_input_layer,
            size_layer_one,
            size_layer_two,
            size_layer_three,
            size_latent_space,
        )

    def forward(self, x):
        """

        :param x: Data tensor
        :return: Returns original data representation
        """
        z = self.encoder(x)

        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    """
    Class containing the whole variational autoencoder.
    """

    def __init__(
        self,
        size_input_layer: int,
        size_layer_one: int,
        size_layer_two: int,
        size_layer_three: int,
        size_latent_space: int,
    ):
        """Constructor"""
        super().__init__()

        self.encoder = VariationalEncoder(
            size_input_layer,
            size_layer_one,
            size_layer_two,
            size_layer_three,
            size_latent_space,
        )
        self.decoder = Decoder(
            size_input_layer,
            size_layer_one,
            size_layer_two,
            size_layer_three,
            size_latent_space,
        )

    def forward(self, x):
        """

        :param x: Data tensor
        :return: Returns original data representation
        """
        z, mu, sigma = self.encoder(x)

        return self.decoder(z), mu, sigma
