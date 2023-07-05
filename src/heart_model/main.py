# Third-party library imports
import torch

# Local import
import src.heart_model.data_handler as dh
import src.heart_model.data_analyzer as da


def main():
    # file = '/media/sf_Share/global_raw.h5ad'
    file = '/home/ubuntu/Projects/Master-Internship/data/global_raw_5000x5000_sample.h5ad'
    path = "h"
    if path == "h":
        handler = dh.Handler(file_location=file)
        handler.get_donors()
    elif path == "a":
        analyzer = da.Analyzer(file_location=file)
        analyzer.pca_analysis()

    # Recognizes if cuda gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # vae = m.VariationalAutoencoder(data_tensor.size()).to(device)


if __name__ == '__main__':
    main()
