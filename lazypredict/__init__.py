# -*- coding: utf-8 -*-

"""
Lazy Predict helps build a lot of basic models without much code and helps understand 
which models work better without any parameter tuning.
"""

__author__ = """Shankar Rao Pandala"""
__email__ = "shankar.pandala@live.com"
__version__ = "0.3.0"  # Increment version for refactored codebase

from .models import (
    LazyClassifier,
    LazyRegressor,
    LazyOrdinalRegressor,
    LazySurvivalAnalysis,
    LazySequencePredictor,
)

__all__ = [
    "LazyClassifier",
    "LazyRegressor",
    "LazyOrdinalRegressor",
    "LazySurvivalAnalysis",
    "LazySequencePredictor",
]
