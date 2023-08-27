# Standard imports
import argparse

# Local imports
import src.constants as const
import src.handler
import src.analyst


def main(arguments):
    # Create a Handler object
    handler = src.handler.Handler(arguments.data)

    if arguments.command == "subset":
        handler.subset_adata(export_path=arguments.export,
                             number_rows=arguments.nrows,
                             number_cols=arguments.ncols,
                             shuffle=arguments.no_shuffle)

    if arguments.command == "analyst":
        handler.data_analysis(arguments.statistics, arguments.pca, arguments.svd)



if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(prog=const.PROGRAM_NAME, description=const.PROGRAM_DESCRIPTION)

    # Define arguments
    parser.add_argument("--data", required=True, help="Path to data file")

    # Create subparser
    subparsers = parser.add_subparsers(description="command", help="Sub-command help")

    # Subset subparser
    parser_subset = subparsers.add_parser("subset", help="Subsets the given dataset")
    parser_subset.add_argument("--export", help="Defines export path and filename for the output")
    parser_subset.add_argument("--nrows", type=int, help="Number of rows used for the subset")
    parser_subset.add_argument("--ncols", type=int, help="Number of columns used for the subset")
    parser_subset.add_argument("--no-shuffle", action="store_false", help="Turns off shuffling of rows")

    # Analyst subparser
    parser_analyst = subparsers.add_parser(name="analyst", help="Provides options for dataset analysis")
    parser_analyst.add_argument(name_or_flags="--statistics", action="store_true",
                                help="Activates statistical analysis")
    parser_analyst.add_argument(name_or_flags="--pca", action="store_true", help="Activates PCA analysis")
    parser.add_argument(name_or_flags="--svd", action="store_true", help="Activates SVD analysis")

    # Parse the arguments
    args = parser.parse_args()
    main(args)
