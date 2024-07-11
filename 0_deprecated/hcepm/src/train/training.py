"""
TRAIN module - TRAINING
With this class, a given model can be trained according to a given set of hyperparameters.
"""

# Standard library imports
from datetime import datetime
from random import choice
from typing import Tuple

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
from . import model


class Trainer:
    def __init__(
        self,
        adata: AnnData,
        model: Tuple[model.Autoencoder, model.VariationalAutoencoder],
        hyperparameters: list[float],
    ):
        """Constructor:

        Docstring
        """
        self.adata = adata
        self.model = model
        self.hyperparams = hyperparameters

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cuda = True if self.device == "cuda" else False

        self.model.to(self.device)
