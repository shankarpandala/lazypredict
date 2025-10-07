# -*- coding: utf-8 -*-

"""Top-level package for Lazy Predict."""

__author__ = """Shankar Rao Pandala"""
__email__ = "shankar.pandala@live.com"
__version__ = '0.2.16'

# Import main classes
from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Import explainer (optional, requires SHAP)
try:
    from lazypredict.Explainer import ModelExplainer
    __all__ = ['LazyClassifier', 'LazyRegressor', 'ModelExplainer']
except ImportError:
    __all__ = ['LazyClassifier', 'LazyRegressor']
