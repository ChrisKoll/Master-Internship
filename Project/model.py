

import torch
from torch import nn
import matplotlib.pyplot as plt


class VariationalAutoencoder(nn.Module):

    def __init__(self, size_input_vector: int, size_latent_space: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(size_input_vector, 20000),
            nn.ReLU(),
            nn.Linear(20000, 10000),
            nn.ReLU(),
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, size_latent_space),
            nn.ReLU()
        )

    def encode(self):
        pass

    def decode(self):
        pass

    def forward(self):
        pass
