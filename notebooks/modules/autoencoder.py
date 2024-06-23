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
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
    """

    z_sample: torch.Tensor
    x_recon: torch.Tensor

    loss: torch.Tensor


# == Autoencoder ==


class Autoencoder(nn.Module):
    """
    Autoencoder (AE) class.

    Args:
        size_input (int): Dimensionality of the input data.
    """

    def __init__(
        self,
        size_input: int,
        size_layers: list[tuple[int, nn.Module]],
        optimizer: nn.Module,
    ):
        super(Autoencoder, self).__init__()  # Why?

        # Input size for later processing
        self.input_size = size_input

        # Encoder architecture
        self.encoder = nn.ModuleList()
        for size, activation in size_layers:
            self.encoder.append(nn.Linear(size_input, size))s
            size_input = size

            # Checks if activation is viable
            if activation is not None:
                assert isinstance(
                    activation, nn.Module
                ), f"Activation should be of type {nn.Module}"

                self.encoder.append(activation)

        # Decoder archtitecture
        self.decoder = nn.ModuleList()
        for size, activation in reversed(size_layers):
            self.decoder.append(nn.Linear(size, size_input))


    def encode(self, x):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Input data compressed to latent space.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)

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
        loss_recon = (
            f.binary_cross_entropy(recon_x, x + 0.5, reduction="none").sum(-1).mean()
        )

        return AEOutput(z_sample=z, x_recon=recon_x, loss=loss_recon)
