"""
Machine learning models for lazypredict.
"""
from .classification import LazyClassifier
from .regression import LazyRegressor
from .survival import LazySurvivalAnalysis
from .sequence import LazySequencePredictor

__all__ = [
    "LazyClassifier",
    "LazyRegressor",
    "LazySurvivalAnalysis",
    "LazySequencePredictor",
]