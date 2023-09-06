# == Standard ==
import argparse

# == Local imports ==
import constants as const
from handler import Handler


def main(arguments):
    # Create a Handler object
    handler = Handler(arguments.data)


if __name__ == '__main__':
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(prog=const.PROGRAM_NAME, description=const.PROGRAM_DESCRIPTION)

    # Define arguments
    parser.add_argument("--data", required=True, help="Path to data file")

    # Create subparser
    subparsers = parser.add_subparsers(dest="command")

    # Parse the arguments
    args = parser.parse_args()
    main(args)
