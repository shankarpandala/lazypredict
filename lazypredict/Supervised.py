"""
Supervised learning module for backward compatibility.

This module provides backward compatibility with older versions of lazypredict.
"""

# Import from new structure
from .models.classification import LazyClassifier
from .models.regression import LazyRegressor
from .models.ordinal import LazyOrdinalRegressor
from .models.survival import LazySurvivalAnalysis
from .models.sequence import LazySequencePredictor

__all__ = [
    "LazyClassifier",
    "LazyRegressor",
    "LazyOrdinalRegressor",
    "LazySurvivalAnalysis",
    "LazySequencePredictor",
]
