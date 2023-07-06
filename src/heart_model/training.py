# Standard library imports
from random import choice
from typing import Optional

# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local import
import src.heart_model.model as mod


class Training:

    def __int__(self, adata: np.matrix, vae: mod.VariationalAutoencoder, donors: Optional[list[str]] = None):
        self.adata = adata
        self.vae = vae
        self.donors = donors

    def create_fold(self, idx: int, *, batch_size: int, learning_rate: float):

        donor = choice(self.donors)

        val_indices = np.where(self.adata[:, idx] == donor)[0]

        # Create dataset and data loaders
        train_dataset = self.adata[~np.isin(self.adata, val_indices)]
        val_dataset = self.adata[val_indices]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model instance
        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        return train_loader, val_loader, optimizer, criterion

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader, optimizer, criterion):
        self.vae.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            recon_batch, mu, log_var = self.vae(batch)

            # Compute reconstruction loss and KL divergence loss
            reconstruction_loss = criterion(recon_batch, batch)
            kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Total loss
            loss = reconstruction_loss + kl_divergence_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_loader)

        # Evaluation on validation set
        self.vae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                recon_batch, mu, log_var = self.vae(batch)
                loss = criterion(recon_batch, batch)
                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(val_loader)

        return train_loss, val_loss

    def train(self, *, epochs: int, batch_size: int, learning_rate: float, device: str):
        # Shuffle the dataset?

        for idx, donor in enumerate(self.donors):

            print(f"Fold {idx + 1}/{len(self.donors)}")

            train_loader, val_loader, optimizer, criterion = self.create_fold(idx, batch_size=batch_size,
                                                                              learning_rate=learning_rate)

            # Training loop
            for epoch in range(epochs):

                train_loss, val_loss = self.train_epoch(train_loader, val_loader, optimizer, criterion)

                # Print epoch statistics
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}")
