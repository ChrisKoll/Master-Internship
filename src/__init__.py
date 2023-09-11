# == Local imports ==
import constants as const
from handler import Handler


def main():
    # file = "data/global_raw.h5ad"
    file = "data/global_raw_5000x5000_sample.h5ad"

    # Create a Handler object
    handler = Handler(file_location=file)
    # handler.statistical_analysis()
    handler.normalize()


if __name__ == "__main__":
    main()
