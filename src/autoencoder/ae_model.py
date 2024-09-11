"""
Autoencoder (AE) Implementation.

This module provides an implementation of a standard Autoencoder using PyTorch. 
An Autoencoder is a type of neural network trained to encode input data into 
a lower-dimensional latent space and then decode it back to reconstruct the 
original input. The goal of the Autoencoder is to learn an efficient 
representation (latent space) of the input data.

Classes:
    - AEOutput: A dataclass that holds the output of the Autoencoder model, including the latent
        space representation, reconstructed data, and reconstruction loss.
    - Autoencoder: A class that implements the Autoencoder model with an encoder, decoder, and loss
        computation.

Methods:
    - encode: Encodes the input data into a latent representation (latent space).
    - decode: Decodes the latent space data back into the original input space.
    - forward: Executes the full forward pass, including encoding, decoding, and optional 
        loss computation.
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

    This dataclass contains the relevant outputs of the Autoencoder's forward pass.

    Attributes:
        z_sample (torch.Tensor): The latent space representation of the input data.
        x_recon (torch.Tensor): The reconstructed output from the Autoencoder.
        loss (torch.Tensor): The computed reconstruction loss (if available).
    """

    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor


class Autoencoder(nn.Module):
    """
    Autoencoder (AE) class.

    The Autoencoder class implements the basic structure of an Autoencoder, consisting
    of an encoder that compresses the input data into a latent space and a decoder
    that reconstructs the input from the latent representation. It also computes
    the reconstruction loss using a specified loss function.

    Attributes:
        encoder (nn.Sequential): A sequential container holding the encoder layers.
        decoder (nn.Sequential): A sequential container holding the decoder layers.
        loss_function (nn.Module): The loss function used for calculating the reconstruction loss.
        device (torch.device): The device (CPU or GPU) on which the model will be executed.
    """

    def __init__(
        self,
        encoder_layers: list[nn.Module],
        decoder_layers: list[nn.Module],
        loss_function: nn.Module,
    ) -> None:
        """
        Initializes the Autoencoder model by setting up the encoder, decoder, and loss function.

        Args:
            encoder_layers (list[nn.Module]): A list of PyTorch layers for the encoder.
            decoder_layers (list[nn.Module]): A list of PyTorch layers for the decoder.
            loss_function (nn.Module): The loss function used to compute reconstruction loss.
        """
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
        Encodes the input data into a lower-dimensional latent space.

        This method passes the input data through the encoder, resulting in a
        compressed latent representation.

        Args:
            x (torch.Tensor): The input data to encode.

        Returns:
            torch.Tensor: The latent space representation of the input data.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent space representation back into the original input space.

        This method takes the compressed latent space representation and passes
        it through the decoder to reconstruct the original data.

        Args:
            z (torch.Tensor): The latent space representation to decode.

        Returns:
            torch.Tensor: The reconstructed output data.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor, compute_loss: bool = True) -> AEOutput:
        """
        Performs a full forward pass of the Autoencoder.

        This method encodes the input data, decodes the latent representation,
        and optionally computes the reconstruction loss. If `compute_loss` is False,
        only the encoded latent space and reconstructed output are returned.

        Args:
            x (torch.Tensor): The input data to be encoded and decoded.
            compute_loss (bool): Whether to compute and return the reconstruction loss.

        Returns:
            AEOutput: A dataclass containing the latent space sample, reconstructed output, and
                optional loss.
        """
        z = self.encode(x)  # Encode input to latent space
        recon_x = self.decode(z)  # Decode latent representation to reconstruct input

        if not compute_loss:
            return AEOutput(z_sample=z, x_recon=recon_x, loss=None)

        # Compute reconstruction loss -> Mean over batch
        loss_recon = self.loss_function(recon_x, x).sum(-1).mean()

        return AEOutput(z_sample=z, x_recon=recon_x, loss=loss_recon)
