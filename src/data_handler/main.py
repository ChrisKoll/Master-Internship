# Standard imports
import argparse

# Local imports
import src.data_handler.data_analyzer as ana

# Create parser
parser = argparse.ArgumentParser()
parser.add_argument("--analyser", action="store_true", help="Activates analysis mode")
parser.add_argument("--trainer", action="store_true", help="Activates training mode")

args = parser.parse_args()

if args.analyser:
    print(ana)