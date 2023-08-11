# Standard library imports
from datetime import datetime
from random import choice

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

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    @staticmethod
    def gaussian_likelihood(x_hat, log_scale, x):
        """
        Docstring
        """
        scale = torch.exp(log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        log_pxz = dist.log_prob(x)

        return log_pxz.sum(dim=(1, 2, 3))

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
        # --> Look at num workers
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_one_epoch(self, train_loader: DataLoader, optimizer):
        """
        Docstring
        """
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            x, labels = data

            # Zero gradients for every batch
            optimizer.zero_grad()

            x_hat, mu, log_var = self.vae(x)

            # Reconstruction loss
            reconstruction_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
            reconstruction_loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                # Loss per batch
                last_loss = running_loss / 1000
                print(f"Batch {i + 1} loss: {last_loss}")
                running_loss = 0.

            return last_loss

    def fit(self, *, epochs: int, batch_size: int, learning_rate: float):
        """
        Docstring
        """
        best_val_loss = 1_000_000.

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        # Iterate over donors
        for idx, _ in enumerate(self.donors):
            print(f"Fold {idx + 1}/{len(self.donors)}")

            train_loader, val_loader = self.create_fold(idx, batch_size=batch_size)

            # Training loop
            for epoch_number in range(epochs):
                self.vae.train(True)
                avg_loss = self.train_one_epoch(train_loader, optimizer=optimizer, loss_function=loss_function)

                running_val_loss = 0.
                self.vae.eval()

                with torch.no_grad():
                    for i, val_data in enumerate(val_loader):
                        val_inputs, val_labels = val_data
                        val_outputs = self.vae(val_inputs)
                        val_loss = loss_function(val_outputs, val_labels)
                        running_val_loss += val_loss

                avg_val_loss = running_val_loss / (i + 1)

                print(f"Loss: training {avg_loss}, validation {avg_val_loss}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    timestamp = datetime.now().strftime("%Y%m%a_%H%M%S")
                    model_path = f"model_{timestamp}_{epoch_number + 1}"
                    torch.save(self.vae.state_dict(), model_path)
