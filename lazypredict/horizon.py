"""Multi-step forecast horizon strategies.

Provides direct and multi-output forecasting as alternatives to
the default recursive (autoregressive) approach.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from lazypredict.ts_preprocessing import create_lag_features

logger = logging.getLogger("lazypredict")


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
) -> np.ndarray:
    """Direct multi-step forecasting: train one model per horizon step.

    Each model h (h=1..horizon) is trained to predict y_{t+h} directly
    from features at time t. More accurate for long horizons but trains
    ``horizon`` separate models.

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
        Exogenous features for training period.
    random_state : int
        Random seed.
    use_gpu : bool
        Whether to use GPU.
    model_params : dict or None
        Extra model constructor params.

    Returns
    -------
    np.ndarray
        Array of length ``horizon`` with direct forecasts.
    """
    from lazypredict.config import get_gpu_model_params

    X_feat, y_feat = create_lag_features(y_train, n_lags, n_rolling, X_exog=X_exog)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    predictions = np.zeros(horizon)

    for h in range(horizon):
        # Target shifted by h steps
        if h == 0:
            y_h = y_feat
            X_h = X_scaled
        else:
            # Need to shift y_feat forward by h
            if len(y_feat) <= h:
                break
            y_h = y_feat[h:]
            X_h = X_scaled[:len(y_h)]

        params = get_gpu_model_params(estimator_class, use_gpu)
        if model_params:
            params.update(model_params)
        try:
            if "random_state" in estimator_class().get_params():
                params["random_state"] = random_state
        except Exception:
            pass

        model = estimator_class(**params)
        model.fit(X_h, y_h)

        # Predict using the last available features
        last_features = X_scaled[-1:] if len(X_scaled) > 0 else X_h[-1:]
        predictions[h] = model.predict(last_features)[0]

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
) -> np.ndarray:
    """Multi-output forecasting: predict all horizon steps at once.

    Uses sklearn's MultiOutputRegressor to wrap the base estimator
    and predict all steps simultaneously from a single feature set.

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
        Exogenous features.
    random_state : int
        Random seed.
    use_gpu : bool
        Whether to use GPU.
    model_params : dict or None
        Extra model constructor params.

    Returns
    -------
    np.ndarray
        Array of length ``horizon``.
    """
    from lazypredict.config import get_gpu_model_params

    X_feat, y_feat = create_lag_features(y_train, n_lags, n_rolling, X_exog=X_exog)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # Build multi-output targets: y[t+1], y[t+2], ..., y[t+horizon]
    n_samples = len(y_feat) - horizon
    if n_samples < 1:
        logger.warning("Not enough data for multi-output forecast with horizon=%d", horizon)
        return np.full(horizon, y_train[-1])

    Y_multi = np.column_stack([
        y_feat[h:n_samples + h] for h in range(1, horizon + 1)
    ])
    X_multi = X_scaled[:n_samples]

    params = get_gpu_model_params(estimator_class, use_gpu)
    if model_params:
        params.update(model_params)
    try:
        if "random_state" in estimator_class().get_params():
            params["random_state"] = random_state
    except Exception:
        pass

    base_model = estimator_class(**params)
    multi_model = MultiOutputRegressor(base_model)
    multi_model.fit(X_multi, Y_multi)

    # Predict from last feature set
    last_features = X_scaled[n_samples - 1:n_samples]
    predictions = multi_model.predict(last_features)[0]

    return predictions[:horizon]
