# Third-party library imports
import torch

# Local/application-specific import
import model


def train(vae: model.VariationalAutoencoder, data: torch.Tensor, *, epochs: int, device: str) -> \
        model.VariationalAutoencoder:
    """

    :param vae: Model architecture to be trained
    :param data: Data the model will be trained on
    :param epochs: Number of epochs
    :param device: Provides the device for training
    :return: Returns the trained VAE model
    """
    # Optimizer
    opt = torch.optim.Adam(vae.parameters())

    for epoch in range(epochs):
        for x in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat) ** 2).sum() + vae.encoder.kullback_leibler_divergence
            loss.backward()
            opt.step()

    return vae
