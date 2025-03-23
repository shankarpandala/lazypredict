"""
Metrics utilities for lazypredict.
"""
import logging
import numpy as np
from typing import Dict, Optional, Union, Callable, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("lazypredict.metrics")

def adjusted_rsquared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """Calculate adjusted R-squared score.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    n_features : int
        Number of features used in the model
        
    Returns
    -------
    float
        Adjusted R-squared score
    """
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate adjusted R-squared
    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adj_r2

def get_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: Optional[int] = None,
    custom_metric: Optional[Callable] = None,
) -> Dict[str, float]:
    """Calculate regression metrics.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    n_features : int, optional (default=None)
        Number of features used for prediction.
        If provided, adjusted R-squared will be calculated.
    custom_metric : callable, optional (default=None)
        Custom metric function that takes y_true and y_pred as arguments
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    metrics = {
        'R-Squared': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
    }
    
    # Add adjusted R-squared if n_features is provided
    if n_features is not None:
        metrics['Adjusted R-Squared'] = adjusted_rsquared(y_true, y_pred, n_features)
    
    # Add custom metric if provided
    if custom_metric is not None:
        try:
            metrics['Custom Metric'] = custom_metric(y_true, y_pred)
        except Exception:
            metrics['Custom Metric'] = np.nan
    
    return metrics

def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    custom_metric: Optional[Callable] = None,
) -> Dict[str, float]:
    """Calculate classification metrics.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    y_proba : array-like of shape (n_samples, n_classes), optional (default=None)
        Probability estimates.
    custom_metric : callable, optional (default=None)
        Custom metric function that takes y_true and y_pred as arguments
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Add ROC AUC if probabilities are available
    if y_proba is not None:
        try:
            # Handle multi-class case
            if y_proba.shape[1] > 2:
                metrics['ROC AUC'] = roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class='ovr',
                    average='weighted'
                )
            else:
                metrics['ROC AUC'] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            # Skip ROC AUC if calculation fails
            pass
    
    # Add custom metric if provided
    if custom_metric is not None:
        try:
            metrics['Custom Metric'] = custom_metric(y_true, y_pred)
        except Exception:
            metrics['Custom Metric'] = np.nan
    
    return metrics