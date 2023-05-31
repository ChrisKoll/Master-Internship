# Third-party library imports
import torch

# Local/application-specific import
import model


def train(vae: model.VariationalAutoencoder, data: torch.Tensor, epochs: int) -> model.VariationalAutoencoder:
    """

    :param vae: Model architecture to be trained
    :param data: Data the model will be trained on
    :param epochs: Number of epochs
    :return: Returns the trained VAE model
    """
    # Recognizes if cuda gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
