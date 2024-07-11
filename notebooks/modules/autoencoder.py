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
        size_layers (list): Dimensionality of the layers.
        loss_function (nn.Module): Loss function used for evaluation.
        optimizer (nn.Module): Optimizer used
    """

    def __init__(
        self, encoder_layers: list, decoder_layers: list, loss_function: nn.Module
    ):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.loss_function = loss_function

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
            AEOutput: VAE output dataclass.
        """
        z = self.encode(x)
        recon_x = self.decode(z)

        if not compute_loss:
            return AEOutput(z_sample=z, x_recon=recon_x, loss=None)

        # compute loss terms
        loss_recon = self.loss_function(recon_x, x)

        return AEOutput(z_sample=z, x_recon=recon_x, loss=loss_recon)
