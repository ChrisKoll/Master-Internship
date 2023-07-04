# Standard library imports
from typing import Optional

# Third-party library imports
from anndata import AnnData

# Local/application-specific import
from src import model


def train(data: AnnData, vae: Optional[model.VariationalAutoencoder] = None, donors: Optional[list[str]] = None,
          *, epochs: int, device: str):
    """
    :param vae: Model architecture to be trained
    :param data: Data the model will be trained on
    :param donors: List of all donors
    :param epochs: Number of epochs
    :param device: Provides the device for training
    :return: Returns the trained VAE model
    """
    # Shuffle the dataset ?

    for fold, donor in enumerate(donors):
        print(f"Fold {fold + 1}/{len(donors)}")

        donor = choice(donors)

        # Split data into train and validation sets based on the current donor
        train_indices = np.where(data[:, donor_index] != donor)[0]
        val_indices = np.where(gene_expression[:, donor_index] == donor)[0]

        # Create dataset and data loaders
        train_dataset = GeneExpressionDataset(gene_expression[train_indices])
        val_dataset = GeneExpressionDataset(gene_expression[val_indices])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model instance
        model = VAE(input_dim, latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()