"""Docstring."""

# Standard imports
from dataclasses import dataclass

__author__ = "Christian Kolland"
__version__ = 1.0


@dataclass
class OptimizerConfig:
    """Docstring."""

    optimizer: str
    learning_rate: float
    weight_decay: float
