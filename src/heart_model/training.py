# Standard library imports
from random import choice
from typing import Optional

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local import
import src.heart_model.model as mod


class Trainer:

    def __init__(self, adata: torch.Tensor, vae: mod.VariationalAutoencoder, donors: list[str]):
        """Constructor:
        
        Docstring
        """
        self.adata = adata
        self.vae = vae
        self.donors = donors

    @staticmethod
    def kullback_leibler_divergence(z, mu, std):
        """
        Docstring
        """
        # Assume normal distributions
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # Kullback-Leibler
        kl = log_qzx - log_pz
        kl = kl.sum(-1)

        return kl

    @staticmethod
    def gaussian_likelihood(x_hat, locscale, x):
        scale = torch.exp(locscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        log_qxz = dist.log_prob(x)

        return log_qxz.sum(dim=(1, 2, 3))

    def create_fold(self, idx: int, *, batch_size: int):
        """
        Docstring
        """
        # Random choice for donor
        donor = choice(self.donors)
        # Compute the indices of all samples in the validation set
        val_indices = np.where(self.adata[:, idx] == donor)[0]

        # Subset dataset
        train_dataset = self.adata[~np.isin(self.adata, val_indices)]
        val_dataset = self.adata[val_indices]

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_step(self, train_loader: DataLoader, val_loader: DataLoader, optimizer, criterion):
        """
        Docstring
        """
        self.vae.train()
        train_loss = 0.0

        for batch in train_loader:

            recon_batch, mu, log_var = self.vae(batch).encoder

            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            x_hat = self.vae()

            # Compute reconstruction loss and KL divergence loss
            reconstruction_loss = criterion(recon_batch, batch)
            kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Total loss
            # loss = reconstruction_loss + kl_divergence_loss

            # Backpropagation and optimization
            reconstruction_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += reconstruction_loss.item()

        # Compute average training loss
        # train_loss /= len(train_loader)

        # Evaluation on validation set
        self.vae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                recon_batch = self.vae(batch)
                loss = criterion(recon_batch, batch)
                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(val_loader)

        return train_loss, val_loss

    def train(self, *, epochs: int, batch_size: int, learning_rate: float):
        """
        Docstring
        """
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Iterate over donors
        for idx, _ in enumerate(self.donors):

            print(f"Fold {idx + 1}/{len(self.donors)}")

            train_loader, val_loader = self.create_fold(idx, batch_size=batch_size)

            # Training loop
            for epoch in range(epochs):

                train_loss, val_loss = self.train_epoch(train_loader, val_loader, optimizer, criterion)

                # Print epoch statistics
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}")
