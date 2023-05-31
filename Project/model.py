# Standard library imports
from typing import Optional

# Third-party library imports
import torch
import torch.nn as nn
import torch.nn.functional as f


# Layer size
# 33538 (input size) * 1/3
SIZE_LAYER_ONE = 10062
SIZE_LAYER_TWO = 3018
SIZE_LAYER_THREE = 906
SIZE_LATENT_SPACE = 200


class VariationalEncoder(nn.Module):
    """
    Class containing the variational encoder.
    """

    def __init__(self, input_dims: int):
        """Constructor

        :param input_dims: Dimension of input vector
        """
        super().__init__()
        self.kullback_leibler_divergence = None
        self.normal_distribution = torch.distributions.Normal(0, 1)
        # Hack to get sampling on the GPU
        self.normal_distribution.loc = self.normal_distribution.loc.cuda()
        self.normal_distribution.scale = self.normal_distribution.scale.cuda()

        # Model architecture
        self.input_layer = nn.Linear(input_dims, SIZE_LAYER_ONE)
        self.hidden_layer1 = nn.Linear(SIZE_LAYER_ONE, SIZE_LAYER_TWO)
        self.hidden_layer2 = nn.Linear(SIZE_LAYER_TWO, SIZE_LAYER_THREE)
        # Latent space for mean and standard deviation
        self.latent_space1 = nn.Linear(SIZE_LAYER_THREE, SIZE_LATENT_SPACE)
        self.latent_space2 = nn.Linear(SIZE_LAYER_THREE, SIZE_LATENT_SPACE)

    def forward(self, x):
        """
        Forward function for the encoder model.

        :param x: Data tensor
        :return: Returns latent space z
        """
        # Reduce input tensor to 1 dimension
        x = torch.flatten(x, start_dim=1)
        # ReLu activation
        x = f.relu(self.input_layer(x))
        x = f.relu(self.hidden_layer1(x))
        x = f.relu(self.hidden_layer2(x))

        # Mean vector
        mu = self.latent_space1(x)
        # Vector of standard deviation
        # Exponential activation
        sigma = torch.exp(self.latent_space2(x))
        # Combined latent space
        z = mu + sigma * self.N.sample(mu.shape)

        # Adjust Kullback-Leibler
        self.kullback_leibler_divergence = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        return z


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
        self.input_layer = nn.Linear(SIZE_LATENT_SPACE, SIZE_LAYER_THREE)
        self.hidden_layer1 = nn.Linear(SIZE_LAYER_THREE, SIZE_LAYER_TWO)
        self.hidden_layer2 = nn.Linear(SIZE_LAYER_TWO, SIZE_LAYER_ONE)
        self.output_layer = nn.Linear(SIZE_LAYER_ONE, output_dims)

    def forward(self, z, *, number_samples, number_genes):
        """
        Forward function for the decoder model.

        :param z: Latent space representation
        :param number_samples: Number of samples for output reshape
        :param number_genes: Number of genes for output reshape
        :return: Original representation of the data
        """
        # ReLu activation
        z = f.relu(self.input_layer(z))
        z = f.relu(self.hidden_layer1(z))
        z = f.relu(self.hidden_layer2(z))
        z = f.relu(self.output_layer(z))

        return z.reshape((number_samples, number_genes))


class VariationalAutoencoder(nn.Module):
    """
    Class containing the whole variational autoencoder.
    """

    def __init__(self, size_input_vector: int, *, device: Optional[str] = None):
        """Constructor

        :param size_input_vector: Size of the input vector
        """
        super().__init__()

        self.input_size = size_input_vector
        self.encoder = VariationalEncoder(size_input_vector)
        self.decoder = Decoder(size_input_vector)

        # Device
        self.device = device

    def forward(self, x):
        """

        :param x: Data tensor
        :return: Returns original data representation
        """
        z = self.encoder(x)

        return self.decoder(z)
