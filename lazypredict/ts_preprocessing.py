"""Time series feature engineering utilities for LazyForecaster."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("lazypredict")


# Seasonal-period detection settings, chosen on simulated seasonal and
# non-seasonal series (white noise, AR(1), random walks, curved trends).
_SEASONAL_Z = 2.5  # ACF peaks must exceed 2.5 short-term standard errors
_SEASONAL_ALPHA = 0.001  # family-wise level of the phase ANOVA
_SEASONAL_MIN_STRENGTH = 0.2  # variance share the seasonal pattern must explain
_SEASONAL_MIN_CYCLES = 3


def detect_seasonal_period(
    y: np.ndarray, max_period: int = 365
) -> Optional[int]:
    """Auto-detect the dominant seasonal period.

    Candidates are local peaks of the autocorrelation function (ACF) of the
    linearly detrended series that exceed 2.5 times Bartlett's standard error
    of its short-term autocorrelation (the lags before the ACF first reaches
    zero), so the seasonal pattern itself does not raise the bar. Each
    candidate is moved to the nearby lag where the values grouped by phase
    differ most, then confirmed by a one-way ANOVA of the values grouped by
    phase (Bonferroni-corrected over all ACF peaks) whose seasonal pattern
    must explain at least 20% of the variance. Of the confirmed periods, the
    shortest that is at least 85% as strong as the strongest is returned, so
    the fundamental period wins over its multiples. When the levels show no
    seasonality, the first differences are searched as well, which finds
    seasonality on top of a random walk.

    Detection needs at least three full cycles; for shorter series pass
    ``seasonal_period`` to ``LazyForecaster`` instead.

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
    scale = max(1.0, float(np.dot(y, y)))
    t = np.linspace(-1.0, 1.0, n)
    detrended = y - np.polyval(np.polyfit(t, y, 1), t)
    for series in (detrended, np.diff(y)):
        series = series - series.mean()
        if np.dot(series, series) <= 1e-12 * scale:
            continue
        period = _seasonal_period_of(series, max_period)
        if period is not None:
            return period
    return None


def _phase_anova(x: np.ndarray, period: int) -> Tuple[float, float]:
    """p-value and adjusted effect size of a one-way ANOVA of *x* grouped by phase.

    *x* must be centred. The effect size is epsilon squared, the share of the
    variance explained by the phase means beyond what chance would give.
    """
    n = len(x)
    phase = np.arange(n) % period
    counts = np.bincount(phase, minlength=period)
    means = np.bincount(phase, weights=x, minlength=period) / counts
    total = float(np.dot(x, x))
    between = float(np.dot(counts, means ** 2))
    within = total - between
    if within <= 1e-12 * total:
        return 0.0, 1.0
    ms_within = within / (n - period)
    statistic = between / (period - 1) / ms_within
    p_value = float(f_distribution.sf(statistic, period - 1, n - period))
    return p_value, (between - (period - 1) * ms_within) / total


def _seasonal_period_of(x: np.ndarray, max_period: int) -> Optional[int]:
    """Seasonal period of the centred, trend-free series *x*, or None."""
    n = len(x)
    max_lag = min(max_period, n // _SEASONAL_MIN_CYCLES)
    if max_lag < 3:
        return None
    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.dot(x[lag:], x[:-lag]) for lag in lags]) / np.dot(x, x)
    peaks = [
        i
        for i in range(1, max_lag - 1)
        if acf[i] > 0 and acf[i] > acf[i - 1] and acf[i] >= acf[i + 1]
    ]
    if not peaks:
        return None
    nonpositive = np.flatnonzero(acf <= 0)
    short_term = acf[: nonpositive[0] if len(nonpositive) else max_lag]
    band = _SEASONAL_Z * np.sqrt((1.0 + 2.0 * np.sum(short_term ** 2)) / n)

    confirmed = {}
    for i in peaks:
        if acf[i] <= band:
            continue
        # The ACF peak of a long period is flat, so noise easily moves it by a
        # lag or two; settle on the nearby lag whose phase groups differ most.
        lag = int(lags[i])
        delta = max(1, lag // 10)
        scored = [
            (period, *_phase_anova(x, period))
            for period in range(max(2, lag - delta), min(max_lag, lag + delta) + 1)
        ]
        period, p_value, strength = max(scored, key=lambda item: item[2])
        if p_value < _SEASONAL_ALPHA / len(peaks) and strength >= _SEASONAL_MIN_STRENGTH:
            confirmed[period] = strength
    if not confirmed:
        return None
    strongest = max(confirmed.values())
    return min(period for period, strength in confirmed.items() if strength >= 0.85 * strongest)


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
    n_rolling = tuple(dict.fromkeys(n_rolling or (3, 7)))  # equal windows count once
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

    # Append the exogenous variables of the rows that were kept
    if X_exog is not None:
        X_exog = np.asarray(X_exog)
        if X_exog.ndim == 1:
            X_exog = X_exog.reshape(-1, 1)
        if len(X_exog) != len(y):
            raise ValueError(
                f"X_exog has {len(X_exog)} rows but the series has {len(y)} values"
            )
        X_features = np.hstack([X_features, X_exog[df.index.to_numpy()]])

    return X_features, y_target


def next_step_features(
    history,
    n_lags: int,
    n_rolling: Optional[Tuple[int, ...]] = None,
    exog_row: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feature row for the step right after *history*.

    Mirrors :func:`create_lag_features`: the row built from ``y[:t]`` here is
    identical to the training row of time ``t``.
    """
    n_rolling = tuple(dict.fromkeys(n_rolling or (3, 7)))
    features: List[float] = [history[-lag] for lag in range(1, n_lags + 1)]
    for window in n_rolling:
        recent = history[-window:]
        features.append(float(np.mean(recent)))
        # ddof=1 matches pandas' rolling().std() used for the training features.
        features.append(float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0)
    features.append(history[-1] - history[-2] if len(history) >= 2 else 0.0)
    if exog_row is not None:
        features.extend(np.asarray(exog_row, dtype=float).ravel().tolist())
    return np.array(features, dtype=float).reshape(1, -1)


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
    predictions: List[float] = []
    history = list(y_history)

    if X_exog is not None:
        X_exog = np.asarray(X_exog)
        if X_exog.ndim == 1:
            X_exog = X_exog.reshape(-1, 1)
        if len(X_exog) < horizon:
            raise ValueError(
                f"X_exog has {len(X_exog)} rows but {horizon} forecast steps were requested"
            )

    for step in range(horizon):
        exog_row = X_exog[step] if X_exog is not None else None
        X_step = scaler.transform(next_step_features(history, n_lags, n_rolling, exog_row))
        pred = float(estimator.predict(X_step)[0])
        predictions.append(pred)
        history.append(pred)

    return np.array(predictions)
