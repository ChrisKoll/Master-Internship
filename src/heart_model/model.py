# Standard library imports
from typing import Optional

# Third-party library imports
import torch
import torch.nn as nn
import torch.nn.functional as f

# Local imports
import src.heart_model.constants as c


class VariationalEncoder(nn.Module):
    """
    Class containing the variational encoder.
    """

    def __init__(self, input_dims: int):
        """Constructor

        :param input_dims: Dimension of input vector
        """
        super().__init__()

        # Model architecture
        self.input_layer = nn.Linear(input_dims, c.SIZE_LAYER_ONE)
        self.hidden_layer1 = nn.Linear(c.SIZE_LAYER_ONE, c.SIZE_LAYER_TWO)
        self.hidden_layer2 = nn.Linear(c.SIZE_LAYER_TWO, c.SIZE_LAYER_THREE)
        # Latent space for mean and standard deviation
        self.latent_space1 = nn.Linear(c.SIZE_LAYER_THREE, c.SIZE_LATENT_SPACE)
        self.latent_space2 = nn.Linear(c.SIZE_LAYER_THREE, c.SIZE_LATENT_SPACE)

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

    def __init__(self, output_dims):
        """Constructor

        :param output_dims: Dimension of output vector
        """
        super().__init__()

        # Model architecture
        self.input_layer = nn.Linear(c.SIZE_LATENT_SPACE, c.SIZE_LAYER_THREE)
        self.hidden_layer1 = nn.Linear(c.SIZE_LAYER_THREE, c.SIZE_LAYER_TWO)
        self.hidden_layer2 = nn.Linear(c.SIZE_LAYER_TWO, c.SIZE_LAYER_ONE)
        self.output_layer = nn.Linear(c.SIZE_LAYER_ONE, output_dims)

    def forward(self, z, *, number_samples, number_genes):
        """
        Forward function for the decoder model.

        :param z: Latent space representation
        :param number_samples: Number of samples for output reshape
        :param number_genes: Number of genes for output reshape
        :return: Original representation of the data
        """
        # ReLU activation
        output = f.relu(self.input_layer(z))
        output = f.relu(self.hidden_layer1(output))
        output = f.relu(self.hidden_layer2(output))
        output = self.output_layer(output)

        return output.reshape((number_samples, number_genes))


class VariationalAutoencoder(nn.Module):
    """
    Class containing the whole variational autoencoder.
    """

    def __init__(self, size_input_vector: int):
        """Constructor

        :param size_input_vector: Size of the input vector
        """
        super().__init__()

        self.input_size = size_input_vector
        self.encoder = VariationalEncoder(size_input_vector)
        self.decoder = Decoder(size_input_vector)

    def forward(self, x):
        """

        :param x: Data tensor
        :return: Returns original data representation
        """
        z, mu, sigma = self.encoder(x)

        return self.decoder(z), mu, sigma
