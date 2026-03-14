"""Ensemble methods for combining top-k forecaster predictions.

Supports simple averaging, inverse-error weighted averaging, and stacking.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("lazypredict")


def ensemble_simple_average(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    """Average predictions from multiple forecasters.

    Parameters
    ----------
    predictions : dict
        Mapping of model name to prediction array.

    Returns
    -------
    np.ndarray
        Averaged predictions.
    """
    if not predictions:
        raise ValueError("No predictions to ensemble.")
    arrays = list(predictions.values())
    return np.mean(arrays, axis=0)


def ensemble_weighted_average(
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, float],
) -> np.ndarray:
    """Inverse-error weighted average of predictions.

    Models with lower error get higher weight.

    Parameters
    ----------
    predictions : dict
        Mapping of model name to prediction array.
    scores : dict
        Mapping of model name to error metric (lower is better).

    Returns
    -------
    np.ndarray
        Weighted-average predictions.
    """
    if not predictions:
        raise ValueError("No predictions to ensemble.")

    names = list(predictions.keys())
    arrays = np.array([predictions[n] for n in names])
    error_vals = np.array([scores.get(n, 1.0) for n in names])

    # Inverse error weights (add epsilon to avoid division by zero)
    eps = 1e-10
    inv_errors = 1.0 / (error_vals + eps)
    weights = inv_errors / inv_errors.sum()

    return np.average(arrays, axis=0, weights=weights)


def ensemble_stacking(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    pred_test: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Stack predictions using a Ridge meta-learner.

    Fits a Ridge regression on the training predictions (predictions)
    to learn optimal combination weights, then applies to test predictions.

    Parameters
    ----------
    predictions : dict
        Mapping of model name to in-sample (train fold) predictions.
    y_true : np.ndarray
        True values corresponding to the predictions.
    pred_test : dict or None
        Mapping of model name to test predictions. If None, returns
        the in-sample stacked predictions.

    Returns
    -------
    np.ndarray
        Stacked predictions.
    """
    from sklearn.linear_model import Ridge

    if not predictions:
        raise ValueError("No predictions to ensemble.")

    names = sorted(predictions.keys())
    X_stack = np.column_stack([predictions[n] for n in names])

    meta = Ridge(alpha=1.0)
    meta.fit(X_stack, y_true)

    if pred_test is not None:
        X_test_stack = np.column_stack([pred_test[n] for n in names])
        return meta.predict(X_test_stack)

    return meta.predict(X_stack)
