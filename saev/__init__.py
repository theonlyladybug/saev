"""
saev is a Python package for training sparse autoencoders (SAEs) on vision transformers (ViTs) in PyTorch.

The main entrypoint to the package is in [main.py](https://github.com/samuelstevens/saev/blob/main/main.py); use `python main.py --help` to see the options and documentation for the script.
"""

from .config import Config

__all__ = ["Config"]
