""" Main module that runs all the code.
"""

# Built-in libraries
from datetime import datetime

# Third-party libraries
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter

# Self-build modules
import modules.autoencoder as ae
from modules.sparse_dataset import SparseDataset
import modules.training as T
import modules.utils as utils

__author__ = "Christian Kolland"
__version__ = "0.0.1"


def main():
    file_path = "../config/autoencoder_test.json"

    model_params = utils.import_model_params(file_path)

    model_architecture = model_params["model"]
    model_training = model_params["training"]

    # Batch size
    batch_size = model_training[
        "batch_size"
    ]  # Power of 2 is optimized in many libraries

    # Training
    num_epochs = model_training["training_epochs"]

    ## Established the type of device used for model processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda = True if device == "cuda" else False

    file_path = "../data/adata_normalized_sample.h5ad"
    # file_path = "../data/adata_30kx10k_normalized_sample.h5ad"

    adata = ad.read_h5ad(filename=file_path)

    count_data = adata.layers["min_max_normalized"]


if __name__ == "__main__":
    main()
