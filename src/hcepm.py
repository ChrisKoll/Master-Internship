# Standard libs
import argparse
from typing import Optional
from typing import Sequence

# Local imports
# ? Right way to import?
from handler import Handler


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="HCEPM")
    parser.add_argument("anndata", metavar="F", help="AnnData file of scRNA-seq data")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parser to handle subsetting of data
    subset_parser = subparsers.add_parser("subset")
    subset_parser.add_argument(
        "-r",
        "--rows",
        type=int,
        help="number of rows included in subset",
    )
    subset_parser.add_argument(
        "-c",
        "--cols",
        type=int,
        help="number of columns included in subset",
    )
    subset_parser.add_argument(
        "-n",
        "--no-shuffle",
        action="store_true",
        help="turns off shuffeling before subsetting data",
    )

    # Parser to handle normalizing data
    norm_parser = subparsers.add_parser("normalize")
    norm_parser.add_argument(
        "-m",
        "--method",
        required=True,
        choices=["cpm", "mmn"],
        help="method of normalization",
    )

    # Parser to handle the training process
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "-m", "--model", required=True, help="config file for model structure"
    )
    train_parser.add_argument(
        "-p", "--hyperparams", required=True, help="config file for hyperparameters"
    )
    train_parser.add_argument(
        "-n",
        "--normalization",
        required=True,
        choices=["cpm"],
        help="method of normalization",
    )
    train_parser.add_argument(
        "-s", "--subset", type=int, nargs=2, help="number of rows and columns of subset"
    )

    args = parser.parse_args(argv)
    Handler(vars(args))

    return 0


if __name__ == "__main__":
    exit(main())
