# -*- coding: utf-8 -*-

"""Top-level package for Lazy Predict."""

__author__ = """Shankar Rao Pandala"""
__email__ = "shankar.pandala@live.com"
__version__ = '0.3.0a4'

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
    # New modules
    "explainability",
    "search_spaces",
    "ts_search_spaces",
    "tuning",
    "ts_tuning",
    "ts_visualization",
    "ts_diagnostics",
    "ensemble",
    "horizon",
    "distributed",
    "spark",
]

from lazypredict.Supervised import LazyClassifier, LazyRegressor
from lazypredict._base import LazyEstimator
from lazypredict.TimeSeriesForecasting import LazyForecaster
