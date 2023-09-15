# == Local imports ==
import constants as const
from handler import Handler


def main():
    # file = "data/global_raw.h5ad"
    # file = "data/adata_5000x33538_sample.h5ad"
    file = "data/adata_20000x10000_sample_cpm_minmax.h5ad"

    # Create a Handler object
    handler = Handler(file_location=file)
    # handler.statistical_analysis()
    # handler.cpm_normalize()
    # handler.min_max_normalize()
    handler.train_ae()


if __name__ == "__main__":
    main()
