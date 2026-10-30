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

    Fits a Ridge regression that maps the models' *validation* predictions
    to the true validation values, then applies it to their test
    predictions.

    Parameters
    ----------
    predictions : dict
        Mapping of model name to out-of-sample predictions for a validation
        period (not the period being evaluated).
    y_true : np.ndarray
        True values for that validation period.
    pred_test : dict
        Mapping of model name to the predictions to combine. Required:
        returning the meta-model's fit on the data it was trained on would
        overstate its accuracy.

    Returns
    -------
    np.ndarray
        Stacked predictions for ``pred_test``.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not predictions:
        raise ValueError("No predictions to ensemble.")
    if pred_test is None:
        raise ValueError(
            "pred_test is required: fit the meta-model on validation predictions "
            "and apply it to separate test predictions."
        )

    names = sorted(predictions.keys())
    X_stack = np.column_stack([predictions[n] for n in names])

    # Standardised inputs make the ridge penalty independent of the series'
    # units; on raw values it shrank small-valued series to their mean.
    meta = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    meta.fit(X_stack, y_true)

    X_test_stack = np.column_stack([pred_test[n] for n in names])
    return np.asarray(meta.predict(X_test_stack))
