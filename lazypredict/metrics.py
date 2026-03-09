"""Metric helper functions for LazyPredict."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def adjusted_rsquared(r2: float, n: int, p: int) -> float:
    """Calculate adjusted R-squared.

    Parameters
    ----------
    r2 : float
        R-squared value.
    n : int
        Number of observations.
    p : int
        Number of predictors.

    Returns
    -------
    float
        Adjusted R-squared value.
    """
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


# ---------------------------------------------------------------------------
# Time series forecasting metrics
# ---------------------------------------------------------------------------


def mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Mean Absolute Percentage Error (MAPE).

    Undefined when y_true contains zeros; those entries are excluded.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE as a percentage (0–100+).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.inf
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Symmetric Mean Absolute Percentage Error (SMAPE).

    More balanced than MAPE; handles zeros better.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        SMAPE as a percentage (0–200).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not mask.any():
        return 0.0
    return float(
        np.mean(
            2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
        )
        * 100
    )


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    Scale-free metric relative to the in-sample naive forecast error.
    ``seasonal_period=1`` compares against a random-walk naive forecast;
    values > 1 use seasonal naive.

    Parameters
    ----------
    y_true : array-like
        Actual test values.
    y_pred : array-like
        Predicted values.
    y_train : array-like
        Training series used to compute the naive scaling factor.
    seasonal_period : int
        Seasonal period for the naive baseline (1 = non-seasonal).

    Returns
    -------
    float
        MASE value.  Values < 1 beat the naive baseline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    naive_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.inf
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1,
) -> dict:
    """Compute all standard forecasting metrics at once.

    Parameters
    ----------
    y_true : array-like
        Actual test values.
    y_pred : array-like
        Predicted values.
    y_train : array-like
        Training series (needed for MASE).
    seasonal_period : int
        Seasonal period for MASE baseline.

    Returns
    -------
    dict
        Keys: mae, rmse, r_squared, mape, smape, mase.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r_squared": float(r2_score(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "mase": mean_absolute_scaled_error(
            y_true, y_pred, y_train, seasonal_period
        ),
    }
