"""
Autoencoder implementation.

This module contains an implementation of an Autoencoder using PyTorch.
The Autoencoder consists of an encoder and a decoder, and it is trained
to reconstruct the input data.

Classes:
    AEOutput: Dataclass for Autoencoder output.
    Autoencoder: Class implementing the Autoencoder model.
"""

__author__ = "Christian Kolland"
__version__ = "1.0"

# Standard imports
from dataclasses import dataclass

# Third-party imports
import torch
import torch.nn as nn


@dataclass
class AEOutput:
    """
    Dataclass for Autoencoder output.

    Attributes:
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the Autoencoder.
        loss (torch.Tensor): The overall loss of the Autoencoder.
    """

    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor


class Autoencoder(nn.Module):
    """
    Autoencoder (AE) class.

    Args:
        encoder_layers (list): List of encoder layers.
        decoder_layers (list): List of decoder layers.
        loss_function (nn.Module): Loss function used for evaluation.

    Attributes:
        encoder (nn.Sequential): Sequential container for encoder layers.
        decoder (nn.Sequential): Sequential container for decoder layers.
        loss_function (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to run the model on (CPU or GPU).
    """

    def __init__(
        self,
        encoder_layers: list[nn.Module],
        decoder_layers: list[nn.Module],
        loss_function: nn.Module,
    ) -> None:
        super(Autoencoder, self).__init__()

        # Define encoder and decoder as sequential containers
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # Define used loss function
        self.loss_function = loss_function

        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Input data compressed to latent space.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, compute_loss: bool = True) -> AEOutput:
        """
        Performs a forward pass of the Autoencoder.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            AEOutput: Autoencoder output dataclass.
        """
        z = self.encode(x)  # Encode input to latent space
        recon_x = self.decode(z)  # Decode latent representation to reconstruct input

        if not compute_loss:
            return AEOutput(z_sample=z, x_recon=recon_x, loss=None)

        # Compute reconstruction loss -> Mean over batch
        loss_recon = self.loss_function(recon_x, x).sum(-1).mean()

        return AEOutput(z_sample=z, x_recon=recon_x, loss=loss_recon)
