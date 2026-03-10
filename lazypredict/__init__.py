# -*- coding: utf-8 -*-

"""Top-level package for Lazy Predict."""

__author__ = """Shankar Rao Pandala"""
__email__ = "shankar.pandala@live.com"
__version__ = '0.3.0a3'

__all__ = [
    "LazyClassifier",
    "LazyRegressor",
    "LazyEstimator",
    "LazyForecaster",
    "Supervised",
    "TimeSeriesForecasting",
    "exceptions",
    "preprocessing",
    "metrics",
    "config",
]

from lazypredict.Supervised import LazyClassifier, LazyRegressor
from lazypredict._base import LazyEstimator
from lazypredict.TimeSeriesForecasting import LazyForecaster
