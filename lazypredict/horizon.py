"""Multi-step forecast horizon strategies.

Provides direct and multi-output forecasting as alternatives to
the default recursive (autoregressive) approach.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from lazypredict.ts_preprocessing import create_lag_features, next_step_features

logger = logging.getLogger("lazypredict")


def _estimator_kwargs(estimator_class, random_state: int, use_gpu: bool, model_params: Optional[dict]) -> dict:
    from lazypredict.config import get_gpu_model_params

    params = get_gpu_model_params(estimator_class, use_gpu)
    try:
        if "random_state" in estimator_class().get_params():
            params["random_state"] = random_state
    except Exception:
        pass
    if model_params:
        params.update(model_params)
    return params


def _exog_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _future_exog(X_exog, X_future, horizon: int) -> Optional[np.ndarray]:
    """Exogenous rows of the forecast period (None when there is no ``X_exog``)."""
    if X_exog is None:
        return None
    if X_future is None or len(np.atleast_1d(X_future)) == 0:
        raise ValueError("X_future (exogenous values for the forecast period) is required")
    rows = _exog_matrix(X_future)
    if len(rows) < horizon:
        raise ValueError(f"X_future has {len(rows)} rows; {horizon} are needed for the horizon")
    return rows[:horizon]


def _fit_predict(model, X_train: np.ndarray, y_train: np.ndarray, x_next: np.ndarray):
    scaler = StandardScaler().fit(X_train)
    model.fit(scaler.transform(X_train), y_train)
    return model.predict(scaler.transform(x_next))


def direct_forecast(
    estimator_class,
    y_train: np.ndarray,
    horizon: int,
    n_lags: int = 10,
    n_rolling: Tuple[int, ...] = (3, 7),
    X_exog: Optional[np.ndarray] = None,
    random_state: int = 42,
    use_gpu: bool = False,
    model_params: Optional[dict] = None,
    X_future: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Direct multi-step forecasting: train one model per horizon step.

    Model ``h`` (``h = 1..horizon``) learns to predict ``y[t + h - 1]`` from
    the lag features available at time ``t`` and, with exogenous variables,
    their values at the target time ``t + h - 1``; all models then predict
    from the lags after ``y_train`` and the matching row of ``X_future``.
    More accurate for long horizons but trains ``horizon`` separate models.

    Parameters
    ----------
    estimator_class : type
        Sklearn-compatible regressor class.
    y_train : np.ndarray
        Training time series.
    horizon : int
        Number of future steps.
    n_lags : int
        Number of lag features.
    n_rolling : tuple of int
        Rolling window sizes.
    X_exog : np.ndarray or None
        Exogenous features for the training period.
    random_state : int
        Random seed.
    use_gpu : bool
        Whether to use GPU.
    model_params : dict or None
        Extra model constructor params.
    X_future : np.ndarray or None
        Exogenous features for the forecast period, one row per step
        (required with ``X_exog``).

    Returns
    -------
    np.ndarray
        Array of length ``horizon`` with direct forecasts.

    Raises
    ------
    ValueError
        If ``y_train`` is too short to train a model for every step.
    """
    X_lags, y_feat = create_lag_features(y_train, n_lags, n_rolling)
    n_rows = len(y_feat)
    if n_rows < horizon + 1:
        raise ValueError(
            f"Not enough history for direct forecasting {horizon} steps: "
            f"{n_rows} training rows after feature engineering, need > {horizon}"
        )
    future = _future_exog(X_exog, X_future, horizon)
    exog = _exog_matrix(X_exog) if X_exog is not None else None
    offset = len(y_train) - n_rows  # time index of the first feature row
    params = _estimator_kwargs(estimator_class, random_state, use_gpu, model_params)
    lags_next = next_step_features(y_train, n_lags, n_rolling)

    predictions = np.empty(horizon)
    for h in range(horizon):
        X_h, x_next = X_lags[: n_rows - h], lags_next
        if exog is not None and future is not None:
            X_h = np.hstack([X_h, exog[offset + h: offset + n_rows]])
            x_next = np.hstack([lags_next, future[h: h + 1]])
        predictions[h] = _fit_predict(estimator_class(**params), X_h, y_feat[h:], x_next)[0]
    return predictions


def multi_output_forecast(
    estimator_class,
    y_train: np.ndarray,
    horizon: int,
    n_lags: int = 10,
    n_rolling: Tuple[int, ...] = (3, 7),
    X_exog: Optional[np.ndarray] = None,
    random_state: int = 42,
    use_gpu: bool = False,
    model_params: Optional[dict] = None,
    X_future: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-output forecasting: predict all horizon steps at once.

    Uses sklearn's MultiOutputRegressor to learn ``y[t], ..., y[t + horizon - 1]``
    from the lag features available at time ``t`` and, with exogenous
    variables, all their values from ``t`` to ``t + horizon - 1``; then
    predicts every step from the lags after ``y_train`` and ``X_future``.

    Parameters
    ----------
    estimator_class : type
        Sklearn-compatible regressor class.
    y_train : np.ndarray
        Training time series.
    horizon : int
        Number of future steps.
    n_lags : int
        Number of lag features.
    n_rolling : tuple of int
        Rolling window sizes.
    X_exog : np.ndarray or None
        Exogenous features for the training period.
    random_state : int
        Random seed.
    use_gpu : bool
        Whether to use GPU.
    model_params : dict or None
        Extra model constructor params.
    X_future : np.ndarray or None
        Exogenous features for the forecast period, one row per step
        (required with ``X_exog``).

    Returns
    -------
    np.ndarray
        Array of length ``horizon``.

    Raises
    ------
    ValueError
        If ``y_train`` is too short for the requested horizon.
    """
    X_lags, y_feat = create_lag_features(y_train, n_lags, n_rolling)
    n_samples = len(y_feat) - horizon + 1
    if n_samples < 2:
        raise ValueError(
            f"Not enough history for multi-output forecasting {horizon} steps: "
            f"{len(y_feat)} training rows after feature engineering"
        )
    future = _future_exog(X_exog, X_future, horizon)
    Y_multi = np.column_stack([y_feat[h:n_samples + h] for h in range(horizon)])
    X_rows, x_next = X_lags[:n_samples], next_step_features(y_train, n_lags, n_rolling)
    if X_exog is not None and future is not None:
        # Every step's exogenous values: rows t .. t + horizon - 1.
        exog = _exog_matrix(X_exog)
        offset = len(y_train) - len(y_feat)
        X_rows = np.hstack([X_rows] + [exog[offset + h: offset + h + n_samples] for h in range(horizon)])
        x_next = np.hstack([x_next, future.reshape(1, -1)])
    params = _estimator_kwargs(estimator_class, random_state, use_gpu, model_params)
    multi_model = MultiOutputRegressor(estimator_class(**params))
    return np.asarray(_fit_predict(multi_model, X_rows, Y_multi, x_next)[0])
