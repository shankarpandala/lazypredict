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


def _training_rows(y_train, n_lags, n_rolling, X_exog):
    X_feat, y_feat = create_lag_features(y_train, n_lags, n_rolling, X_exog=X_exog)
    scaler = StandardScaler()
    return scaler.fit_transform(X_feat), y_feat, scaler


def _next_row(scaler, y_train, n_lags, n_rolling, X_exog, X_future):
    """Scaled feature row for the first step after ``y_train``."""
    exog_row = None
    if X_exog is not None:
        if X_future is None or len(np.atleast_1d(X_future)) == 0:
            raise ValueError("X_future (exogenous values for the forecast period) is required")
        exog_row = np.atleast_2d(np.asarray(X_future, dtype=float))[0]
    return scaler.transform(next_step_features(y_train, n_lags, n_rolling, exog_row))


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
    the features available at time ``t``; all models then predict from the
    features of the first step after ``y_train``. More accurate for long
    horizons but trains ``horizon`` separate models.

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
        Exogenous features for the forecast period (required with
        ``X_exog``; its first row is used).

    Returns
    -------
    np.ndarray
        Array of length ``horizon`` with direct forecasts.

    Raises
    ------
    ValueError
        If ``y_train`` is too short to train a model for every step.
    """
    X_scaled, y_feat, scaler = _training_rows(y_train, n_lags, n_rolling, X_exog)
    if len(y_feat) < horizon + 1:
        raise ValueError(
            f"Not enough history for direct forecasting {horizon} steps: "
            f"{len(y_feat)} training rows after feature engineering, need > {horizon}"
        )
    params = _estimator_kwargs(estimator_class, random_state, use_gpu, model_params)
    x_next = _next_row(scaler, y_train, n_lags, n_rolling, X_exog, X_future)

    predictions = np.empty(horizon)
    for h in range(horizon):
        model = estimator_class(**params)
        model.fit(X_scaled[:len(y_feat) - h], y_feat[h:])
        predictions[h] = model.predict(x_next)[0]
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
    from the features available at time ``t``, then predicts all steps from
    the features of the first step after ``y_train``.

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
        Exogenous features for the forecast period (required with
        ``X_exog``; its first row is used).

    Returns
    -------
    np.ndarray
        Array of length ``horizon``.

    Raises
    ------
    ValueError
        If ``y_train`` is too short for the requested horizon.
    """
    X_scaled, y_feat, scaler = _training_rows(y_train, n_lags, n_rolling, X_exog)
    n_samples = len(y_feat) - horizon + 1
    if n_samples < 2:
        raise ValueError(
            f"Not enough history for multi-output forecasting {horizon} steps: "
            f"{len(y_feat)} training rows after feature engineering"
        )
    Y_multi = np.column_stack([y_feat[h:n_samples + h] for h in range(horizon)])
    params = _estimator_kwargs(estimator_class, random_state, use_gpu, model_params)
    multi_model = MultiOutputRegressor(estimator_class(**params))
    multi_model.fit(X_scaled[:n_samples], Y_multi)

    x_next = _next_row(scaler, y_train, n_lags, n_rolling, X_exog, X_future)
    return np.asarray(multi_model.predict(x_next)[0])
