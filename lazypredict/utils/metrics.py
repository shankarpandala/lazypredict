"""
Metrics utilities for lazypredict.
"""
import logging
import numpy as np
from typing import Dict, Optional, Union
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

def get_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate regression metrics.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics["R-Squared"] = r2_score(y_true, y_pred)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    
    # Additional metrics
    metrics["Explained Variance"] = explained_variance_score(y_true, y_pred)
    metrics["Max Error"] = max_error(y_true, y_pred)
    metrics["Median Absolute Error"] = median_absolute_error(y_true, y_pred)
    
    return metrics

def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
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
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    # Try to calculate ROC AUC if probabilities are available
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) <= 2:  # Binary classification
                if y_proba.shape[1] == 2:
                    metrics["ROC AUC"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)
            else:  # Multi-class
                metrics["ROC AUC"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception as e:
            logger.warning(f"Error calculating ROC AUC: {e}")
            metrics["ROC AUC"] = np.nan
    
    # F1, precision, recall (weighted for multi-class)
    try:
        metrics["F1 Score"] = f1_score(y_true, y_pred, average="weighted")
        metrics["Precision"] = precision_score(y_true, y_pred, average="weighted")
        metrics["Recall"] = recall_score(y_true, y_pred, average="weighted")
    except Exception as e:
        logger.warning(f"Error calculating F1/precision/recall: {e}")
        metrics["F1 Score"] = np.nan
        metrics["Precision"] = np.nan
        metrics["Recall"] = np.nan
    
    return metrics 