# Standard imports
import argparse

# Third-party library imports

# Local import


def main():
    """
    # Recognizes if cuda gpu is available
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    file = "/home/ubuntu/Projects/Master-Internship/data/normalized_global_raw_5000x5000_sample.h5ad"
    handler = dh.Handler(file_location=file)

    donors = handler.get_donors()
    print(donors)
    input_data = handler.to_tensor()
    # No cuda
    # input_data.to(device)

    # vae = mod.VariationalAutoencoder(c.SIZE_INPUT_LAYER).to(device)
    vae = mod.VariationalAutoencoder(c.SIZE_INPUT_LAYER)
    training = train.Training(input_data, vae, donors)
    training.train(batch_size=c.BATCH_SIZE, epochs=c.NUM_EPOCHS, learning_rate=c.LEARNING_RATE)
    """


if __name__ == '__main__':
    main()
