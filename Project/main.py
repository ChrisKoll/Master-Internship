# Third-party library imports
import torch

# Local/application-specific import
import data_handler as dh
import model as m
import training as t


def main():
    file = '/media/sf_Share/sample_data.h5ad'
    handler = dh.DataHandler(file)
    data_tensor = handler.to_tensor()

    # Recognizes if cuda gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = m.VariationalAutoencoder(data_tensor.size()).to(device)
    vae = t.train(vae, data_tensor, epochs=20, device=device)


if __name__ == '__main__':
    main()
