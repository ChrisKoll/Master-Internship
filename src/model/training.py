# Standard library imports
from datetime import datetime
from random import choice

# Third-party library imports
from anndata import AnnData
from anndata.experimental.pytorch import AnnLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision import transforms

# Local import
import constants as const
import model.model as mod


class Trainer:
    def __init__(self, adata, model):
        """Constructor:

        Docstring
        """
        self.adata = adata
        self.model = model

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda = True if self.device == "cuda" else False

        self.model.to(self.device)

    def create_fold(self, *, batch_size: int):
        """
        Docstring
        """
        # Random choice for donor
        donor = choice(self.adata.obs["donor"].unique())

        # Subset dataset
        train_dataset = self.adata[self.adata.obs["donor"] != donor]
        val_dataset = self.adata[self.adata.obs["donor"] == donor]

        # Create dataloaders
        # --> Look at num workers
        train_loader = AnnLoader(
            train_dataset, batch_size=batch_size, shuffle=True, use_cuda=self.cuda
        )
        val_loader = AnnLoader(
            val_dataset, batch_size=batch_size, shuffle=False, use_cuda=self.cuda
        )

        return train_loader, val_loader

    def train_one_epoch(self, train_loader, loss_function, optimizer):
        """
        Docstring
        """
        epoch_loss = []
        for idx, batch in enumerate(train_loader):
            # Reconstruct input
            x_hat = self.model(batch.layers["minmax_normalized"])

            # Calculate loss
            loss = loss_function(x_hat, batch.layers["minmax_normalized"])

            # Zero gradients for every batch
            optimizer.zero_grad()
            loss.backward()
            # Adjust learning weights
            optimizer.step()

            # Loss per batch
            epoch_loss.append(loss.item())
            print(f"Batch {idx + 1}/{len(train_loader)} loss: {loss.item()}")

        return epoch_loss

    def fit_ae(
        self, *, epochs: int, batch_size: int, learning_rate: float, weight_decay: float
    ):
        """
        Docstring
        """
        total_training_loss = {}
        total_validation_loss = {}

        # Define optimizer and loss function
        loss_function = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Iterate over donors
        for idx, _ in enumerate(self.adata.obs["donor"].unique()):
            print(f"Fold {idx + 1}/{len(self.adata.obs['donor'].unique())}")

            train_loader, val_loader = self.create_fold(batch_size=batch_size)

            # Training loop
            for epoch_number in range(epochs):
                print(f"Epoch {epoch_number + 1}/{epochs}")
                self.model.train()
                training_loss = self.train_one_epoch(
                    train_loader, loss_function=loss_function, optimizer=optimizer
                )

                running_val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    validaton_loss = []
                    for idx, val_batch in enumerate(val_loader):
                        val_outputs = self.model(val_batch.layers["minmax_normalized"])
                        val_loss = loss_function(
                            val_outputs, val_batch.layers["minmax_normalized"]
                        )
                        validaton_loss.append(val_loss)
                        print(
                            f"Batch {idx + 1}/{len(val_loader)} validation loss: {val_loss.item()}"
                        )

                total_training_loss[f"Epoch {epoch_number}"] = training_loss
                total_validation_loss[f"Epoch {epoch_number}"] = validaton_loss

        return total_training_loss, total_validation_loss

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

    def train_one_epoch(self, train_loader: DataLoader, optimizer):
        """
        Docstring
        """
        running_loss = 0.0
        last_loss = 0.0

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
            running_loss += reconstruction_loss.item()
            if i % 1000 == 999:
                # Loss per batch
                last_loss = running_loss / 1000
                print(f"Batch {i + 1} loss: {last_loss}")
                running_loss = 0.0

            return last_loss

    def fit_vae(
        self, *, epochs: int, batch_size: int, learning_rate: float, weight_decay: float
    ):
        """
        Docstring
        """
        total_training_loss = {}
        total_validation_loss = {}

        # Define optimizer and loss function
        loss_function = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Iterate over donors
        for idx, _ in enumerate(self.adata.obs["donor"].unique()):
            print(f"Fold {idx + 1}/{len(self.adata.obs['donor'].unique())}")

            train_loader, val_loader = self.create_fold(batch_size=batch_size)

            # Training loop
            for epoch_number in range(epochs):
                print(f"Epoch {epoch_number}/{epochs}")
                self.model.train()
                training_loss = self.train_one_epoch(
                    train_loader, loss_function=loss_function, optimizer=optimizer
                )

                running_val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    validaton_loss = []
                    for idx, val_batch in enumerate(val_loader):
                        val_outputs = self.model(val_batch.layers["minmax_normalized"])
                        val_loss = loss_function(
                            val_outputs, val_batch.layers["minmax_normalized"]
                        )
                        validaton_loss.append(val_loss)
                        print(
                            f"Batch {idx}/{len(val_loader)} validation loss: {val_loss.item()}"
                        )

                total_training_loss[f"Epoch {epoch_number}"] = training_loss
                total_validation_loss[f"Epoch {epoch_number}"] = validaton_loss

        return total_training_loss, total_validation_loss
