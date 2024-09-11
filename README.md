# Cell Expression Reconstruction with AE/VAE

Internship project - Bioinformatics M.Sc.

This project implements both Autoencoder (AE) and Variational Autoencoder (VAE) models using PyTorch. The models, along with necessary utility functions and configurations, are designed to train on scRNA-seq data from the [Heart Cell Atlas](https://www.heartcellatlas.org/) with a flexible, modular architecture.

## Usage

The primary scripts for training the models are train_vae.py and train_ae.py. Both scripts accept several command-line arguments to specify the dataset, configuration, and logging options.

### Arguments

| Flag | Description | Example |
|:---:|:---:|:---:|
| -d, --data | Path to the dataset file. | path/to/data.h5ad |
| -l, --layer	| Name of the data layer used for fitting. | min_max_scaled |
| -c, --conf | Path to the configuration JSON file. | configs/default_config.json |
| -x, --log | Directory where logs will be saved. Default is logs/. | logs/ |
| -f, --name | Optional name for the log file. | vae_training.log |

