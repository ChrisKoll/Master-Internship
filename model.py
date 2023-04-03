import torch
from torch import nn
import torch.nn.functional as f


# Input img -> Hidden dim -> mean, std -> Parameterization trick -> Decoder -> Output img
class VariationalAutoencoder(nn.Module):

    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

    def encode(self):
        pass

    def decode(self):
        pass

    def forward(self):
        pass


def main():
    # 28*28
    x = torch.randn(1, 784)
    vae = VariationalAutoencoder()
    print(vae(x).shape)


