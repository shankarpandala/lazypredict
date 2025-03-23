"""Top-level package for lazypredict."""

__author__ = """Shankar Rao Pandala"""
__email__ = "shankar.pandala@live.com"
__version__ = "0.2.12"

import warnings
warnings.filterwarnings("ignore")

from .Supervised import Supervised, LazyClassifier, LazyRegressor

__all__ = [
    "Supervised",
    "LazyClassifier",
    "LazyRegressor",
]
