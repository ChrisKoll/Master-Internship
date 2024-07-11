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
        size_layers (list): Dimensionality of the layers.
        loss_function (nn.Module): Loss function used for evaluation.
        optimizer (nn.Module): Optimizer used
    """

    def __init__(
        self,
        size_layers: list[tuple[int, nn.Module]],
        loss_function,
        optimizer,
        learning_rate=1e-2,
        weight_decay=1e-3,
    ):
        super(Autoencoder, self).__init__()

        self.size_layers = size_layers
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.loss_function = loss_function
        self.optimizer = optimizer(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _build_encoder(self) -> nn.Sequential:
        """
        Builds the encoder architecture

        Args:
            size_layers (list(tuple)): Encoder Architecture
        """
        ## Encoder architecture
        encoder_layers = []
        # Only iterate until second to last element
        # --> Idx of last element called
        for idx, (size, activation) in enumerate(self.size_layers[:-1]):
            # While second to last element no reached
            # --> Activation function in decoder
            if idx < len(self.size_layers[:-1]) - 1:
                encoder_layers.append(nn.Linear(size, self.size_layers[idx + 1][0]))

                # Checks if activation is viable
                if activation is not None:
                    assert isinstance(
                        activation, nn.Module
                    ), f"Activation should be of type {nn.Module}"
                    encoder_layers.append(activation)
            else:
                encoder_layers.append(nn.Linear(size, self.size_layers[idx + 1][0]))

        print("Constructed encoder...")

        return nn.Sequential(*encoder_layers)

    def _build_decoder(self) -> nn.Sequential:
        """
        Builds the decoder architecture

        Args:
            size_layers (list(tuple)): Decoder Architecture
        """
        # Reverse to build decoder (hourglass)
        reversed_layers = list(reversed(self.size_layers))
        decoder_layers = []
        for idx, (size, activation) in enumerate(reversed_layers[:-1]):
            # While second to last element no reached
            # --> Activation function in encoder
            if idx < len(reversed_layers[:-1]) - 1:
                decoder_layers.append(nn.Linear(size, reversed_layers[idx + 1][0]))

                # Checks if activation is viable
                if activation is not None:
                    assert isinstance(
                        activation, nn.Module
                    ), f"Activation should be of type {nn.Module}"
                    decoder_layers.append(activation)
            else:
                decoder_layers.append(nn.Linear(size, reversed_layers[idx + 1][0]))

        print("Constructed decoder...")

        return nn.Sequential(*decoder_layers)

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
        loss_recon = self.criterion(recon_x, x)

        return AEOutput(z_sample=z, x_recon=recon_x, loss=loss_recon)
