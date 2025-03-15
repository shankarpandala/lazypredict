"""
Supervised learning module for lazypredict.

This module provides high-level interfaces for supervised learning tasks.
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .models import (
    LazyClassifier,
    LazyRegressor,
    LazyOrdinalRegressor,
    LazySurvivalAnalysis,
    LazySequencePredictor,
)

def get_card_split(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data ensuring all classes are represented in both train and test sets.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split
    random_state : int, optional (default=None)
        Random state for reproducibility
        
    Returns
    -------
    X_train : array-like of shape (n_train_samples, n_features)
        Training data
    X_test : array-like of shape (n_test_samples, n_features)
        Test data
    y_train : array-like of shape (n_train_samples,)
        Training labels
    y_test : array-like of shape (n_test_samples,)
        Test labels
    """
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_samples = len(cls_indices)
        n_test = int(test_size * n_samples)
        
        if n_test == 0:
            n_test = 1
        elif n_test == n_samples:
            n_test = n_samples - 1
            
        # Shuffle indices
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            rng.shuffle(cls_indices)
        else:
            np.random.shuffle(cls_indices)
            
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def adjusted_rsquared(r2: float, n_samples: int, n_features: int) -> float:
    """Calculate adjusted R-squared.
    
    Parameters
    ----------
    r2 : float
        R-squared value
    n_samples : int
        Number of samples
    n_features : int
        Number of features
        
    Returns
    -------
    float
        Adjusted R-squared value
    """
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# Define exports
__all__ = [
    'LazyClassifier',
    'LazyRegressor',
    'LazyOrdinalRegressor',
    'LazySurvivalAnalysis',
    'LazySequencePredictor',
    'get_card_split',
    'adjusted_rsquared',
]

# Issue deprecation warning for old import structure
warnings.warn(
    "Importing directly from lazypredict.Supervised is deprecated. "
    "Please use 'from lazypredict.models import LazyClassifier, LazyRegressor' instead.",
    DeprecationWarning,
    stacklevel=2,
)
