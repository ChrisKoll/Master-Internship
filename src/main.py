# == Standard ==
import argparse

# == Local imports ==
import constants as const
from handler import Handler


def main(arguments):
    # Create a Handler object
    handler = Handler(arguments.data)

    if arguments.command == "subset":
        handler.subset_adata(export_path=arguments.export,
                             number_rows=arguments.nrows,
                             number_cols=arguments.ncols,
                             shuffle=arguments.no_shuffle)


if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(prog=const.PROGRAM_NAME, description=const.PROGRAM_DESCRIPTION)

    # Define arguments
    parser.add_argument("--data", required=True, help="Path to data file")

    # Create subparser
    subparsers = parser.add_subparsers(dest="command")

    # Subset subparser
    parser_subset = subparsers.add_parser("subset", help="Subset given dataset")
    parser_subset.add_argument("--export", help="Defines export path and filename for the output")
    parser_subset.add_argument("--nrows", type=int, help="Number of rows used for the subset")
    parser_subset.add_argument("--ncols", type=int, help="Number of columns used for the subset")
    parser_subset.add_argument("--no-shuffle", action="store_false", help="Turns off shuffling of rows")

    # Parse the arguments
    args = parser.parse_args()
    main(args)
