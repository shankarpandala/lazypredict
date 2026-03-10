"""Hyperparameter tuning engine for LazyForecaster.

Provides Optuna-based tuning for time series forecasting models with
forecasting-specific features: tunable n_lags/n_rolling, seasonal period
search, and temporal cross-validation.
"""

import copy
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from lazypredict.metrics import compute_forecast_metrics
from lazypredict.ts_search_spaces import get_ts_search_space

logger = logging.getLogger("lazypredict")

# Optional Optuna
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# Valid forecast metrics (lower is better for all except R-Squared)
VALID_TUNE_METRICS = {"RMSE", "MAE", "MAPE", "SMAPE", "MASE"}
CANDIDATE_SEASONAL_PERIODS = [7, 12, 24, 52, 365]


def tune_forecaster_optuna(
    model_name: str,
    wrapper,
    y_train: np.ndarray,
    X_train: Optional[np.ndarray],
    seasonal_period: Optional[int],
    tune_metric: str = "RMSE",
    cv: int = 3,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    random_state: int = 42,
    tune_seasonal: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Tune a single forecasting model using Optuna.

    Parameters
    ----------
    model_name : str
        Name of the forecaster.
    wrapper : ForecasterWrapper
        The forecaster wrapper instance (will be deep-copied per trial).
    y_train : np.ndarray
        Training time series.
    X_train : np.ndarray or None
        Exogenous features.
    seasonal_period : int or None
        Detected seasonal period.
    tune_metric : str
        Metric to optimize ('RMSE', 'MAE', 'MAPE', 'SMAPE', 'MASE').
    cv : int
        Number of TimeSeriesSplit folds.
    n_trials : int
        Number of Optuna trials.
    timeout : float or None
        Seconds limit per model tuning.
    random_state : int
        Random seed.
    tune_seasonal : bool
        If True, also search over seasonal_period values.

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best metric score (lower is better).
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna is required for tuning. Install with: pip install lazypredict[tune]"
        )

    space_fn = get_ts_search_space(model_name)
    if space_fn is None:
        logger.info("No search space for %s, skipping tuning.", model_name)
        return {}, float("inf")

    metric_key = tune_metric.lower()
    tscv = TimeSeriesSplit(n_splits=cv)

    def objective(trial):
        params = space_fn(trial)

        # Optional seasonal period search
        if tune_seasonal and model_name in ("SARIMAX", "HoltWinters_Add", "HoltWinters_Mul"):
            candidates = [p for p in CANDIDATE_SEASONAL_PERIODS if p < len(y_train) // 3]
            if seasonal_period and seasonal_period not in candidates:
                candidates.append(seasonal_period)
            if candidates:
                params["seasonal_period"] = trial.suggest_categorical(
                    "seasonal_period", sorted(candidates)
                )

        # Extract feature-engineering params for ML/DL models
        n_lags = params.pop("n_lags", None)
        n_rolling_1 = params.pop("n_rolling_1", None)
        n_rolling_2 = params.pop("n_rolling_2", None)
        sp_trial = params.pop("seasonal_period", seasonal_period)

        scores = []
        for train_idx, val_idx in tscv.split(y_train):
            y_cv_train = y_train[train_idx]
            y_cv_val = y_train[val_idx]
            X_cv_train = X_train[train_idx] if X_train is not None else None
            X_cv_val = X_train[val_idx] if X_train is not None else None

            try:
                w = copy.deepcopy(wrapper)
                # Apply tuned params to wrapper
                _apply_params_to_wrapper(
                    w, model_name, params, n_lags, n_rolling_1, n_rolling_2, sp_trial
                )
                w.fit(y_cv_train, X_cv_train)
                y_pred = w.predict(len(val_idx), X_cv_val)
                sp = sp_trial or 1
                fold_metrics = compute_forecast_metrics(
                    y_cv_val, y_pred, y_cv_train, seasonal_period=sp
                )
                scores.append(fold_metrics[metric_key])
            except Exception:
                scores.append(float("inf"))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study.best_params, study.best_value


def _apply_params_to_wrapper(wrapper, model_name, params, n_lags, n_rolling_1, n_rolling_2, sp):
    """Apply tuned parameters to a forecaster wrapper instance."""
    from lazypredict.TimeSeriesForecasting import MLForecaster, _TorchRNNForecaster

    # ML/DL models: update feature-engineering params
    if isinstance(wrapper, (MLForecaster, _TorchRNNForecaster)):
        if n_lags is not None:
            wrapper.n_lags = n_lags
        if n_rolling_1 is not None and n_rolling_2 is not None:
            wrapper.n_rolling = (n_rolling_1, n_rolling_2)

    # Statistical model params applied at fit time via wrapper attributes
    if model_name == "SARIMAX" and "order" in params:
        wrapper.order = params["order"]
    if model_name in ("HoltWinters_Add", "HoltWinters_Mul") and sp is not None:
        wrapper.seasonal_periods = sp

    # DL-specific params
    if isinstance(wrapper, _TorchRNNForecaster):
        for key in ("hidden_size", "learning_rate", "batch_size", "n_epochs"):
            if key in params:
                setattr(wrapper, key, params[key])

    # ML model estimator params
    if isinstance(wrapper, MLForecaster):
        # Store extra model params to be passed at fit time
        model_params = {k: v for k, v in params.items()
                        if k not in ("order", "seasonal_period")}
        if model_params:
            wrapper._tune_params = model_params


def tune_top_k_forecasters(
    scores_df,
    models: Dict[str, Any],
    all_wrappers: List[Tuple[str, Any]],
    y_train: np.ndarray,
    X_train: Optional[np.ndarray],
    seasonal_period: Optional[int],
    top_k: int = 5,
    tune_metric: str = "RMSE",
    cv: int = 3,
    n_trials: int = 30,
    timeout: Optional[float] = None,
    random_state: int = 42,
    tune_seasonal: bool = False,
) -> "pd.DataFrame":
    """Tune the top-k forecasters from initial ranking.

    Returns
    -------
    pd.DataFrame
        Tuning results: Model, Best Score, Best Params.
    """
    import pandas as pd

    top_model_names = list(scores_df.index[:top_k])
    name_to_wrapper = {name: w for name, w in all_wrappers}

    results = []
    for model_name in top_model_names:
        wrapper = name_to_wrapper.get(model_name)
        if wrapper is None:
            continue

        space_fn = get_ts_search_space(model_name)
        if space_fn is None:
            logger.info("No search space for %s, skipping.", model_name)
            results.append({
                "Model": model_name,
                "Best Score": None,
                "Best Params": "N/A (no search space)",
            })
            continue

        logger.info("Tuning forecaster %s...", model_name)
        best_params, best_score = tune_forecaster_optuna(
            model_name, wrapper, y_train, X_train, seasonal_period,
            tune_metric=tune_metric, cv=cv, n_trials=n_trials,
            timeout=timeout, random_state=random_state,
            tune_seasonal=tune_seasonal,
        )
        results.append({
            "Model": model_name,
            f"Best {tune_metric}": best_score,
            "Best Params": str(best_params),
        })

    return pd.DataFrame(results).set_index("Model")
