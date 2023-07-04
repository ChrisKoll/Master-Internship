# Third-party library imports
import torch

# Local/application-specific import
from src.heart_model import data_handler as dh
import training as t


def main():
    file = '/media/sf_Share/sample_data.h5ad'
    handler = dh.Handler(file)
    # data_tensor = handler.to_tensor()

    # Recognizes if cuda gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # vae = m.VariationalAutoencoder(data_tensor.size()).to(device)

    t.train(data=handler.get_adata, epochs=20, device=device)


if __name__ == '__main__':
    main()
