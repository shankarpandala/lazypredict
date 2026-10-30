"""Time series feature engineering utilities for LazyForecaster."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("lazypredict")


def detect_seasonal_period(
    y: np.ndarray, max_period: int = 365
) -> Optional[int]:
    """Auto-detect the dominant seasonal period using autocorrelation.

    The series is linearly detrended (and first-differenced when it behaves
    like a random walk), then a lag is accepted as a seasonal period only if
    its autocorrelation is a local peak that is significant against
    Bartlett's standard error (3 sigma) and rises clearly above the lowest
    autocorrelation at shorter lags. When several lags qualify, the shortest
    one whose autocorrelation is within 85% of the strongest is returned, so
    the fundamental period wins over its multiples.

    Detection needs at least two full cycles and, in practice, a few more;
    for short series pass ``seasonal_period`` to ``LazyForecaster`` instead.

    Parameters
    ----------
    y : np.ndarray
        Time series values.
    max_period : int
        Maximum candidate period to consider.

    Returns
    -------
    int or None
        Detected seasonal period, or ``None`` if no significant
        seasonality is found.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 8 or not np.all(np.isfinite(y)):
        return None

    t = np.arange(n)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    centered = resid - resid.mean()
    var = np.dot(centered, centered) / n
    if var == 0:
        return None
    if np.dot(centered[1:], centered[:-1]) / n / var > 0.95:
        # Random-walk-like: work on the increments instead.
        centered = np.diff(y)
        centered = centered - centered.mean()
        n = len(centered)
        var = np.dot(centered, centered) / n
        if var == 0:
            return None

    max_lag = min(max_period, n // 2)
    if max_lag < 3:
        return None
    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.dot(centered[lag:], centered[:-lag]) for lag in lags]) / n / var

    # Bartlett's standard error of the autocorrelation at each lag.
    cumulative = np.concatenate([[0.0], np.cumsum(acf[:-1] ** 2)])
    band = 3.0 * np.sqrt((1.0 + 2.0 * cumulative) / n)

    candidates = [
        i
        for i in range(1, len(acf) - 1)
        if acf[i] > acf[i - 1]
        and acf[i] >= acf[i + 1]
        and acf[i] > band[i]
        and acf[i] - acf[:i].min() > band[i]
    ]
    if not candidates:
        return None
    strongest = max(acf[i] for i in candidates)
    return int(lags[min(i for i in candidates if acf[i] >= 0.85 * strongest)])


def create_lag_features(
    y: np.ndarray,
    n_lags: int = 10,
    n_rolling: Optional[Tuple[int, ...]] = None,
    X_exog: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create lag features, rolling statistics, and a diff feature.

    Parameters
    ----------
    y : np.ndarray
        Time series values.
    n_lags : int
        Number of lag features (y_{t-1} … y_{t-n_lags}).
    n_rolling : tuple of int or None
        Window sizes for rolling mean and rolling std.
    X_exog : np.ndarray or None
        Exogenous features to append (shape ``(len(y), k)``).

    Returns
    -------
    X_features : np.ndarray
        Feature matrix of shape ``(n_valid, n_features)``.
    y_target : np.ndarray
        Target values aligned with the feature rows.
    """
    n_rolling = n_rolling or (3, 7)
    df = pd.DataFrame({"y": y})

    # Lag features
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Rolling statistics (shifted by 1 to avoid data leakage)
    for window in n_rolling:
        df[f"rolling_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["y"].shift(1).rolling(window).std()

    # First-difference feature
    df["diff_1"] = df["y"].diff().shift(1)

    # Drop rows with NaN
    df = df.dropna()
    y_target = df["y"].values
    X_features = df.drop(columns=["y"]).values

    # Append exogenous variables (trimmed to match)
    if X_exog is not None:
        X_exog = np.asarray(X_exog)
        if X_exog.ndim == 1:
            X_exog = X_exog.reshape(-1, 1)
        X_exog_trimmed = X_exog[-len(y_target):]
        X_features = np.hstack([X_features, X_exog_trimmed])

    return X_features, y_target


def recursive_forecast(
    estimator,
    scaler: StandardScaler,
    y_history: np.ndarray,
    horizon: int,
    n_lags: int,
    n_rolling: Optional[Tuple[int, ...]] = None,
    X_exog: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-step recursive (autoregressive) forecast.

    At each step the model predicts the next value, which is then appended
    to the history before computing features for the following step.

    Parameters
    ----------
    estimator
        Fitted sklearn-compatible regressor.
    scaler : StandardScaler
        Fitted scaler used during training.
    y_history : np.ndarray
        Historical series used for context.
    horizon : int
        Number of future steps to predict.
    n_lags : int
        Number of lag features.
    n_rolling : tuple of int or None
        Rolling window sizes.
    X_exog : np.ndarray or None
        Exogenous features for the forecast period (shape ``(horizon, k)``).

    Returns
    -------
    np.ndarray
        Array of length ``horizon`` with predicted values.
    """
    n_rolling = n_rolling or (3, 7)
    predictions: List[float] = []
    history = list(y_history)

    if X_exog is not None:
        X_exog = np.asarray(X_exog)
        if X_exog.ndim == 1:
            X_exog = X_exog.reshape(-1, 1)

    for step in range(horizon):
        features: List[float] = []

        # Lags
        for lag in range(1, n_lags + 1):
            features.append(history[-lag])

        # Rolling stats
        for window in n_rolling:
            recent = history[-window:]
            features.append(float(np.mean(recent)))
            # ddof=1 matches pandas' rolling().std() used for the training features.
            features.append(float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0)

        # Diff
        features.append(history[-1] - history[-2] if len(history) >= 2 else 0.0)

        # Exogenous
        if X_exog is not None and step < len(X_exog):
            features.extend(X_exog[step].tolist())

        X_step = np.array(features).reshape(1, -1)
        X_step = scaler.transform(X_step)
        pred = float(estimator.predict(X_step)[0])
        predictions.append(pred)
        history.append(pred)

    return np.array(predictions)
