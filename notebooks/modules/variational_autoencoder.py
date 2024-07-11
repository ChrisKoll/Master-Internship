from dataclasses import dataclass

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

    Args:
        size_input_layer (int): Dimensionality of the input data.
        size_layer_one (int): Dimensionality of hidden layer 1.
        size_layer_two (int): Dimensionality of hidden layer 2.
        size_layer_three (int): Dimensionality of hidden layer 3.
        size_latent_space (int): Dimensionality of the latent space.
    """

    def __init__(
        self,
        size_layers: list[tuple[int, nn.Module]],
        loss_function,
        optimizer,
        learning_rate=1e-2,
        weight_decay=1e-3,
    ):
        super(VariationalAutoencoder, self).__init__()

        self.size_layers = size_layers
        self.encoder = self._build_encoder()
        self.softplus = nn.Softplus()
        self.decoder = self._build_decoder()

        self.criterion = loss_function
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
                encoder_layers.append(nn.Linear(size, 2 * self.size_layers[idx + 1][0]))

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

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

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
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            VAEOutput: VAE output dataclass.
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
        loss_recon = (
            f.binary_cross_entropy(recon_x, x + 0.5, reduction="none").sum(-1).mean()
        )
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
