"""
Metrics for evaluating model performance.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def get_classification_metrics(y_true, y_pred, y_score=None):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values
        
    y_pred : array-like
        Predicted target values
        
    y_score : array-like, optional
        Predicted probabilities
        
    Returns
    -------
    metrics : dict
        Dictionary with classification metrics
    """
    metrics = {}
    
    # Always calculate these metrics
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    # Handle binary classification
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        metrics["F1 Score"] = f1_score(y_true, y_pred, average="binary")
        metrics["Precision"] = precision_score(y_true, y_pred, average="binary")
        metrics["Recall"] = recall_score(y_true, y_pred, average="binary")
        
        # Calculate ROC AUC if probability scores are provided
        if y_score is not None:
            try:
                metrics["ROC AUC"] = roc_auc_score(y_true, y_score)
            except:
                metrics["ROC AUC"] = np.nan
    else:
        # For multiclass problems
        metrics["F1 Score"] = f1_score(y_true, y_pred, average="weighted")
        metrics["Precision"] = precision_score(y_true, y_pred, average="weighted")
        metrics["Recall"] = recall_score(y_true, y_pred, average="weighted")
        
        # ROC AUC is more complex for multiclass
        if y_score is not None:
            try:
                metrics["ROC AUC"] = roc_auc_score(y_true, y_score, multi_class="ovr")
            except:
                metrics["ROC AUC"] = np.nan
    
    return metrics

def get_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values
        
    y_pred : array-like
        Predicted target values
        
    Returns
    -------
    metrics : dict
        Dictionary with regression metrics
    """
    metrics = {}
    
    # Calculate regression metrics
    metrics["R-Squared"] = r2_score(y_true, y_pred)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return metrics 