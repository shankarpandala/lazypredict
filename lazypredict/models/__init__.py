"""
Machine learning models for lazypredict.
"""

from .classification import LazyClassifier
from .regression import LazyRegressor
from .sequence import LazySequencePredictor
from .survival import LazySurvivalAnalysis

__all__ = [
    "LazyClassifier",
    "LazyRegressor",
    "LazySurvivalAnalysis",
    "LazySequencePredictor",
]
