"""
Autoencoder Training Module with Outer Cross-Validation.

This module facilitates the training and evaluation of autoencoder models on single-cell data 
using outer cross-validation. It supports different types of autoencoders, such as standard 
autoencoders and variational autoencoders. The module provides functionality for performing 
training, validation, and testing while logging performance metrics via TensorBoard.

Classes:
    - Training: A class to handle the setup, training, validation, and testing of the 
      autoencoder using cross-validation.

Methods:
    - fit: Initializes and manages the training process across cross-validation folds.
    - _train_fold: Handles the model training for each fold, including splitting the data 
      into train, validation, and test sets.
    - _train_epoch: Executes a single epoch of training and computes the gradient norms and 
      loss for each batch.
    - _val_epoch: Validates the model on the validation dataset for a single epoch, 
      calculating validation loss.
    - _test_fold: Tests the model on the test dataset from the current fold and logs performance metrics.
"""

# Standard imports
from datetime import datetime
from logging import Logger
from typing import Optional, Union

# Third-party imports
from anndata import AnnData
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Self-built modules
from src.autoencoder.ae_model import Autoencoder
import src.utils.data_utils as dutils
from src.utils.json_utils import OptimizerConfig
from src.variational_autoencoder.vae_model import VariationalAutoencoder

__author__ = "Christian Kolland"
__version__ = 1.0


class Training:
    """
    A class for training autoencoder models with outer cross-validation.

    This class provides methods for initializing the training process, managing data splits,
    and logging metrics during training, validation, and testing of the model. The training
    is done using an outer cross-validation technique, ensuring that the model is tested on
    completely unseen data.
    """

    def __init__(
        self,
        model: Union[Autoencoder, VariationalAutoencoder],
        model_name: str,
        adata: AnnData,
        data_layer: str,
        optim_config: OptimizerConfig,
        train_config: dutils.TrainingConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initializes the training object with model and dataset details.

        Args:
            model (Union[Autoencoder, VariationalAutoencoder]): The model to be trained.
            model_name (str): The name of the model.
            adata (AnnData): The AnnData object containing the single-cell dataset.
            data_layer (str): The key in `adata.layers` where input data is stored.
            optim_config (OptimizerConfig): Configuration for the optimizer (learning rate, etc.).
            train_config (TrainingConfig): Configuration for training, including batch size and
                                           number of epochs.
            logger (Optional[Logger]): Logger for logging information. Defaults to None.
        """
        self.model = model
        self.model_name = model_name
        self.adata = adata
        self.data_layer = data_layer
        self.optimizer = self._initialize_optimizer(
            optim_config, self.model.parameters()
        )
        self.training = train_config
        self.logger = logger

    @staticmethod
    def _initialize_optimizer(
        optim_config: OptimizerConfig, model_parameters
    ) -> optim.Optimizer:
        """
        Initializes the optimizer using the provided configuration.

        Args:
            optim_config (OptimizerConfig): Configuration containing optimizer settings.
            model_parameters: Model parameters to be optimized.

        Returns:
            optim.Optimizer: Initialized optimizer.
        """
        optimizer = optim_config.optimizer(
            model_parameters,
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
        )

        return optimizer

    def fit(self, use_writer: bool = True) -> None:
        """
        Sets up and runs the training process for the model across cross-validation folds.

        Args:
            use_writer (bool): Whether to use TensorBoard to log training progress. Defaults to True.
        """
        writer = None
        if use_writer:
            writer = SummaryWriter(
                f'runs/hca/{self.model_name}_{datetime.now().strftime("%d%m%Y-%H%M")}'
            )

        self._train_fold(writer)

        # Close properly
        if use_writer:
            writer.close()

    def _train_fold(self, writer: Optional[SummaryWriter] = None) -> None:
        """
        Trains the model across the provided cross-validation folds.

        This function performs data splitting into training, validation, and test sets for
        each fold and then trains the model accordingly.

        Args:
            writer (Optional[SummaryWriter]): TensorBoard writer for logging progress. Defaults to None.
        """
        for fold_idx, fold in enumerate(self.training.folds):
            if self.logger is not None:
                self.logger.info(
                    f">>> FOLD {fold_idx + 1}/{len(self.training.folds)} - {fold}"
                )

            # Create outer fold -> Testing
            # Test data from this donor will not be seen during training
            train_data, test_loader = dutils.create_outer_fold(
                self.adata, fold, self.data_layer, self.training.batch_size, self.logger
            )

            # Split training data -> Training/validation
            # Standard split for training and validation
            train_loader, val_loader = dutils.split_data(
                train_data,
                self.data_layer,
                batch_size=self.training.batch_size,
                logger=self.logger,
            )

            for epoch in range(self.training.num_epochs):
                if self.logger is not None:
                    self.logger.info(
                        f">>> EPOCH {epoch + 1}/{self.training.num_epochs}"
                    )

                # Training and validation
                self._train_epoch(fold, epoch + 1, train_loader, writer)
                self._val_epoch(fold, epoch + 1, val_loader, writer)

            # Test on fold
            # If model performs good on completly unseen data it generalizes good
            # Guarantees that no donor specific batch effects are learned
            self._test_fold(fold, test_loader, writer)

    def _train_epoch(
        self,
        fold: str,
        epoch: int,
        train_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        """
        Trains the model for one epoch.

        This function performs forward and backward passes on the training data and updates the
        model parameters. Gradient norms and losses are logged for each batch.

        Args:
            fold (str): The identifier for the current fold.
            epoch (int): The current epoch number.
            train_loader (DataLoader): DataLoader for the training data.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging. Defaults to None.
        """
        self.model.train()

        total_norm, total_loss = 0.0, 0.0
        # Adds additional loss terms if model is VAE
        if isinstance(self.model, VariationalAutoencoder):
            total_recon_loss, total_kld = 0.0, 0.0

        with tqdm(train_loader, desc="Training") as train_loop:
            for batch_idx, (batch, _) in enumerate(train_loop):
                batch = batch.to(self.training.device)

                outputs = self.model(batch)
                loss = outputs.loss

                self.optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backpropagation

                # Grad norm after batch
                grad_norm = 0
                for p in self.model.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** (1.0 / 2)
                total_norm += grad_norm

                # Loss after batch
                total_loss += outputs.loss
                if isinstance(self.model, VariationalAutoencoder):
                    total_recon_loss += outputs.loss_recon
                    total_kld += outputs.loss_kl

                if batch_idx % 100 == 0:
                    if self.logger is not None:
                        self.logger.info(
                            f"Epoch {epoch} - Batch {batch_idx}: GradNorm: {grad_norm:.4f}, Loss: {loss.item():.4f}"
                        )

                # Clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update the model parameters
                # Needs to be after gradient norm calculation
                self.optimizer.step()

        # Avg grad norm
        avg_norm = total_norm / len(train_loader)

        # Avg training loss
        avg_loss = total_loss / len(train_loader)
        if isinstance(self.model, VariationalAutoencoder):
            avg_recon_loss = total_recon_loss / len(train_loader)
            avg_kld = total_kld / len(train_loader)

        if writer is not None:
            writer.add_scalar(f"{fold}/Train/GradNorm", avg_norm, epoch)
            writer.add_scalar(f"{fold}/Train/Loss", avg_loss, epoch)

            if isinstance(self.model, VariationalAutoencoder):
                writer.add_scalar(f"{fold}/Train/Loss/Recon", avg_recon_loss, epoch)
                writer.add_scalar(f"{fold}/Train/Loss/KLD", avg_kld, epoch)

        if self.logger is not None:
            self.logger.info(
                f">>> Epoch {epoch} - TRAINING: Avg GradNorm: {avg_norm:.4f}, Avg Loss: {avg_loss:.4f}"
            )

            if isinstance(self.model, VariationalAutoencoder):
                self.logger.info(
                    f">>> Epoch {epoch} - TRAINING: Avg ReconLoss: {avg_recon_loss:.4f}, Avg KLD: {avg_kld:.4f}"
                )

    def _val_epoch(
        self,
        fold: str,
        epoch: int,
        val_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        """
        Validates the model on the validation dataset.

        This function computes the validation loss for the given epoch and logs it.

        Args:
            fold (str): The identifier for the current fold.
            epoch (int): The current epoch number.
            val_loader (DataLoader): DataLoader for the validation data.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging. Defaults to None.
        """
        self.model.eval()

        total_loss = 0.0
        # Adds additional loss terms if model is VAE
        if isinstance(self.model, VariationalAutoencoder):
            total_recon_loss, total_kld = 0.0, 0.0

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as val_loop:
                for data, _ in val_loop:
                    data = data.to(self.training.device)

                    outputs = self.model(data)

                    # Loss after batch
                    total_loss += outputs.loss
                    if isinstance(self.model, VariationalAutoencoder):
                        total_recon_loss += outputs.loss_recon
                        total_kld += outputs.loss_kl

        # Avg validation loss
        avg_loss = total_loss / len(val_loader)
        if isinstance(self.model, VariationalAutoencoder):
            avg_recon_loss = total_recon_loss / len(val_loader)
            avg_kld = total_kld / len(val_loader)

        if writer is not None:
            writer.add_scalar(f"{fold}/Val/Loss", avg_loss, epoch)

            if isinstance(self.model, VariationalAutoencoder):
                writer.add_scalar(f"{fold}/Val/Loss/Recon", avg_recon_loss, epoch)
                writer.add_scalar(f"{fold}/Val/Loss/KLD", avg_kld, epoch)

        if self.logger is not None:
            self.logger.info(
                f">>> Epoch {epoch} - VALIDATION: Avg Loss: {avg_loss:.4f}"
            )

            if isinstance(self.model, VariationalAutoencoder):
                self.logger.info(
                    f">>> Epoch {epoch} - VALIDATION: Avg ReconLoss: {avg_recon_loss:.4f}, Avg KLD: {avg_kld:.4f}"
                )

    def _test_fold(
        self,
        fold: str,
        test_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        """
        Tests the model on the test dataset.

        This function computes the test loss for the current fold and logs it.

        Args:
            fold (str): The identifier for the current fold.
            test_loader (DataLoader): DataLoader for the test data.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging. Defaults to None.
        """
        self.model.eval()

        recons, latent_reps = [], []
        with torch.no_grad():
            with tqdm(test_loader, desc="Testing") as test_loop:
                for batch_idx, (batch, labels) in enumerate(test_loop):
                    batch = batch.to(self.training.device)

                    outputs = self.model(batch)

                    if writer is not None:
                        writer.add_scalar(f"{fold}/Test/Loss", outputs.loss, batch_idx)

                        if isinstance(self.model, VariationalAutoencoder):
                            writer.add_scalar(
                                f"{fold}/Test/Loss/Recon", outputs.loss_recon, batch_idx
                            )
                            writer.add_scalar(
                                f"{fold}/Test/Loss/KLD", outputs.loss_kl, batch_idx
                            )

                    if batch_idx % 10 == 0:
                        if self.logger is not None:
                            self.logger.info(
                                f">>> Batch {batch_idx + 1} - TEST: {outputs.loss:.4f}"
                            )

                            if isinstance(self.model, VariationalAutoencoder):
                                self.logger.info(
                                    f">>> Batch {batch_idx + 1} - TEST: ReconLoss: {outputs.loss_recon:.4f}, KLD: {outputs.loss_kl:.4f}"
                                )

                    # Data for performance plotting
                    recons.extend(list(zip(batch, outputs.x_recon, labels)))
                    latent_reps.extend(list(zip(outputs.z_sample, labels)))

        if writer is not None:
            dutils.plot_recon_performance(
                recons,
                dir=self.model_name,
                scope="Sample",
                method="Sum",
                fold=fold,
            )
            dutils.plot_recon_performance(
                recons,
                dir=self.model_name,
                scope="Gene",
                method="Sum",
                fold=fold,
                logger=self.logger,
            )
            dutils.plot_latent_space(latent_reps, dir=self.model_name, fold=fold)
