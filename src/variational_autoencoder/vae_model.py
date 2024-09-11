"""
Variational Autoencoder (VAE) Implementation.

This module defines a Variational Autoencoder (VAE) model, including its forward pass,
reparameterization trick, and loss computation. It is capable of encoding input data into
a latent space, decoding from the latent space to the original space, and computing the 
loss using reconstruction and KL divergence terms.

Classes:
    - VAEOutput: A dataclass that holds the output of the VAE model.
    - VariationalAutoencoder: A class implementing a Variational Autoencoder with an 
      encoder, decoder, and the ability to compute reconstruction and KL divergence loss.

Methods:
    - encode: Encodes the input data into a normal distribution in latent space.
    - reparameterize: Applies the reparameterization trick to sample from the latent space.
    - decode: Decodes the latent space representation into the original input space.
    - forward: Performs the forward pass through the VAE, optionally computing the loss.
"""

__author__ = "Christian Kolland"
__version__ = "1.0"

# Standard imports
from dataclasses import dataclass

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as f


@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.

    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """

    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    This class defines a VAE model with an encoder and a decoder. The encoder compresses
    the input into a latent space, and the decoder reconstructs the input from the latent
    space. The model also computes the VAE loss, which consists of reconstruction loss
    and KL divergence.

    Attributes:
        encoder (nn.Sequential): A sequential container holding the encoder layers.
        softplus (nn.Softplus): Sofplus function from PyTorch
        decoder (nn.Sequential): A sequential container holding the decoder layers.
        loss_function (nn.Module): The loss function used for calculating the reconstruction loss.
        device (torch.device): The device (CPU or GPU) on which the model will be executed.
    """

    def __init__(
        self,
        encoder_layers: list[nn.Module],
        decoder_layers: list[nn.Module],
        loss_function: nn.Module,
    ):
        """
        Initializes the VAE with encoder, decoder, and loss function.

        Args:
            encoder_layers (list[nn.Module]): A list of PyTorch layers for the encoder.
            decoder_layers (list[nn.Module]): A list of PyTorch layers for the decoder.
            loss_function (nn.Module): The loss function used to compute reconstruction loss.
        """
        super(VariationalAutoencoder, self).__init__()

        # Define encoder and decoder as sequential containers
        self.encoder = nn.Sequential(*encoder_layers)
        self.softplus = nn.Softplus()
        self.decoder = nn.Sequential(*decoder_layers)

        # Define used loss function
        self.loss_function = loss_function

        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def encode(self, x, eps: float = 1e-8) -> torch.distributions.MultivariateNormal:
        """
        Encodes the input data into the latent space as a multivariate normal distribution.

        Args:
            x (torch.Tensor): Input data to encode.
            eps (float): A small value to avoid numerical instability in the variance.

        Returns:
            torch.distributions.MultivariateNormal: A normal distribution in the latent space.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist) -> torch.Tensor:
        """
        Reparameterizes the encoded distribution to sample from it.

        The reparameterization trick allows for backpropagation through stochastic nodes
        by separating the randomness from the parameters of the distribution.

        Args:
            dist (torch.distributions.MultivariateNormal): The latent distribution from the encoder.

        Returns:
            torch.Tensor: A sample from the latent distribution.
        """
        return dist.rsample()

    def decode(self, z) -> torch.Tensor:
        """
        Decodes the latent space sample into the original input space.

        Args:
            z (torch.Tensor): Sampled latent space data.

        Returns:
            torch.Tensor: Reconstructed input data.
        """
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True) -> VAEOutput:
        """
        Performs a forward pass of the VAE, encoding the input, sampling from the latent space,
        decoding the latent representation, and optionally computing the VAE loss.

        Args:
            x (torch.Tensor): The input data.
            compute_loss (bool): If True, the loss (reconstruction + KL divergence) is computed.
                If False, only the encoded and decoded data are returned.

        Returns:
            VAEOutput: A dataclass containing the encoded distribution, latent sample,
                reconstructed output, and optionally the losses.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # compute loss terms
        loss_recon = self.loss_function(recon_x, x).sum(-1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device)
            .unsqueeze(0)
            .expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
