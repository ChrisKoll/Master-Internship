"""Autoencoder implementation.

Implementation of an Autoencoder ...
"""

__author__ = "Christian Kolland"
__license__ = "MIT"
__version__ = "0.1"

# == Libraries ==

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as f


# == AE Output ==


@dataclass
class AEOutput:
    """
    Dataclass for AE output.

    Attributes:
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the AE.
        loss (torch.Tensor): The overall loss of the AE.
    """

    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor


# == Autoencoder ==
class Autoencoder(nn.Module):
    """
    Autoencoder (AE) class.

    Args:
        size_layers (list): Dimensionality of the layers.
        loss_function (nn.Module): Loss function used for evaluation.
        optimizer (nn.Module): Optimizer used
    """

    def __init__(
        self,
        size_layers: list[tuple[int, nn.Module]],
        criterion: nn.modules.loss._Loss,
        learning_rate: float = 1e-1,
        weight_decay: float = 1e-8,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
    ):
        super(Autoencoder, self).__init__()

        ## Encoder architecture
        self.encoder_layers = []
        # Only iterate until second to last element
        # --> Idx of last element called
        for idx, (size, activation) in enumerate(size_layers[:-1]):
            # While second to last element no reached
            # --> Activation function in decoder
            if idx < len(size_layers[:-1]) - 1:
                self.encoder_layers.append(nn.Linear(size, size_layers[idx + 1][0]))

                # Checks if activation is viable
                if activation is not None:
                    assert isinstance(
                        activation, nn.Module
                    ), f"Activation should be of type {nn.Module}"
                    self.encoder_layers.append(activation)
            else:
                self.encoder_layers.append(nn.Linear(size, size_layers[idx + 1][0]))

        self.encoder = nn.Sequential(*self.encoder_layers)

        print("Constructed encoder...")

        ## Decoder archtitecture
        # Reverse to build decoder (hourglass)
        reversed_layers = list(reversed(size_layers))
        self.decoder_layers = []
        for idx, (size, activation) in enumerate(reversed_layers[:-1]):
            # While second to last element no reached
            # --> Activation function in encoder
            if idx < len(reversed_layers[:-1]) - 1:
                self.decoder_layers.append(nn.Linear(size, reversed_layers[idx + 1][0]))

                # Checks if activation is viable
                if activation is not None:
                    assert isinstance(
                        activation, nn.Module
                    ), f"Activation should be of type {nn.Module}"
                    self.decoder_layers.append(activation)
            else:
                self.decoder_layers.append(nn.Linear(size, reversed_layers[idx + 1][0]))

        self.decoder = nn.Sequential(*self.decoder_layers)

        print("Constructed decoder...")

        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optimizer = optimizer(
            params=self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode(self, x):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Input data compressed to latent space.
        """
        return x

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return z

    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the AE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            VAEOutput: VAE output dataclass.
        """
        z = self.encode(x)
        recon_x = self.decode(z)

        if not compute_loss:
            return AEOutput(z_sample=z, x_recon=recon_x, loss=None)

        # compute loss terms
        loss_recon = self.criterion(recon_x, x)

        return AEOutput(z_sample=z, x_recon=recon_x, loss=loss_recon)
