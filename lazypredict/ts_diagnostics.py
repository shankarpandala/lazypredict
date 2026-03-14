"""Residual diagnostics for time series forecasting models.

Provides statistical tests and analysis to validate forecast quality:
Ljung-Box test for autocorrelation, Jarque-Bera test for normality,
and ACF computation. Falls back gracefully when statsmodels is unavailable.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("lazypredict")

# Optional statsmodels for statistical tests
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.stattools import acf as sm_acf

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


def residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
    max_lags: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute comprehensive residual diagnostics.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    y_train : array-like or None
        Training data (unused currently, reserved for future MASE-based tests).
    seasonal_period : int
        Seasonal period for selecting ACF lag count.
    max_lags : int or None
        Maximum number of ACF lags. Defaults to ``min(len(residuals)-1, max(10, 2*seasonal_period))``.

    Returns
    -------
    dict
        Keys:
        - ``residuals``: np.ndarray — raw residuals (y_true - y_pred)
        - ``mean``: float — mean of residuals (should be near 0)
        - ``std``: float — standard deviation of residuals
        - ``ljung_box_stat``: float or None — Ljung-Box Q statistic
        - ``ljung_box_pvalue``: float or None — p-value (>0.05 = white noise)
        - ``jarque_bera_stat``: float or None — Jarque-Bera statistic
        - ``jarque_bera_pvalue``: float or None — p-value (>0.05 = normal)
        - ``acf_values``: np.ndarray — autocorrelation values
        - ``is_white_noise``: bool — True if Ljung-Box p > 0.05
        - ``is_normal``: bool — True if Jarque-Bera p > 0.05
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    n = len(residuals)

    if max_lags is None:
        max_lags = min(n - 1, max(10, 2 * seasonal_period))
    max_lags = max(1, min(max_lags, n - 1))

    result: Dict[str, Any] = {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)) if n > 1 else 0.0,
    }
    result["acf_values"] = _compute_acf(residuals, max_lags)
    result.update(_ljung_box(residuals, max_lags))
    result.update(_jarque_bera(residuals))
    return result


def _compute_acf(residuals: np.ndarray, max_lags: int) -> np.ndarray:
    """Compute ACF using statsmodels if available, else numpy fallback."""
    n = len(residuals)
    if _STATSMODELS_AVAILABLE and n > 1:
        try:
            return sm_acf(residuals, nlags=max_lags, fft=True)
        except Exception:
            pass
    return _compute_acf_numpy(residuals, max_lags)


def _ljung_box(residuals: np.ndarray, max_lags: int) -> Dict[str, Any]:
    """Run Ljung-Box test, returning a dict of results."""
    n = len(residuals)
    if _STATSMODELS_AVAILABLE and n > max_lags + 1:
        try:
            lb_result = acorr_ljungbox(residuals, lags=max_lags, return_df=True)
            lb_stat = float(lb_result.iloc[-1]["lb_stat"])
            lb_pvalue = float(lb_result.iloc[-1]["lb_pvalue"])
            return {
                "ljung_box_stat": lb_stat,
                "ljung_box_pvalue": lb_pvalue,
                "is_white_noise": lb_pvalue > 0.05,
            }
        except Exception:
            pass
    return {"ljung_box_stat": None, "ljung_box_pvalue": None, "is_white_noise": None}


def _jarque_bera(residuals: np.ndarray) -> Dict[str, Any]:
    """Run Jarque-Bera test, returning a dict of results."""
    n = len(residuals)
    if _STATSMODELS_AVAILABLE and n >= 8:
        try:
            jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
            return {
                "jarque_bera_stat": float(jb_stat),
                "jarque_bera_pvalue": float(jb_pvalue),
                "is_normal": float(jb_pvalue) > 0.05,
            }
        except Exception:
            pass
    return {"jarque_bera_stat": None, "jarque_bera_pvalue": None, "is_normal": None}


def compare_diagnostics(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
) -> pd.DataFrame:
    """Run residual diagnostics across multiple models.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    predictions : dict
        ``{model_name: y_pred}`` mapping.
    y_train : array-like or None
        Training data.
    seasonal_period : int
        Seasonal period.

    Returns
    -------
    pd.DataFrame
        One row per model with diagnostic statistics.
    """
    if not predictions:
        raise ValueError("No predictions provided.")

    rows = []
    for name, y_pred in predictions.items():
        diag = residual_diagnostics(
            y_true, y_pred, y_train=y_train, seasonal_period=seasonal_period
        )
        rows.append({
            "Model": name,
            "Residual Mean": diag["mean"],
            "Residual Std": diag["std"],
            "Ljung-Box Stat": diag["ljung_box_stat"],
            "Ljung-Box p-value": diag["ljung_box_pvalue"],
            "White Noise": diag["is_white_noise"],
            "Jarque-Bera Stat": diag["jarque_bera_stat"],
            "Jarque-Bera p-value": diag["jarque_bera_pvalue"],
            "Normal": diag["is_normal"],
        })

    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_acf_numpy(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation using numpy (fallback when statsmodels unavailable)."""
    x = x - np.mean(x)
    n = len(x)
    var = np.sum(x ** 2) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.array([
        np.sum(x[: n - k] * x[k:]) / (n * var)
        for k in range(max_lag + 1)
    ])
    return acf
