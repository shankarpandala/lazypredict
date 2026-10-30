"""Hyperparameter tuning engine for LazyForecaster.

Provides Optuna-based tuning for time series forecasting models with
forecasting-specific features: tunable n_lags/n_rolling, seasonal period
search, and temporal cross-validation.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from lazypredict.metrics import compute_forecast_metrics
from lazypredict.ts_search_spaces import get_ts_search_space
from lazypredict._optional import warn_if_broken

logger = logging.getLogger("lazypredict")

# Optional Optuna
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except Exception as exc:  # missing, or installed but broken
    _OPTUNA_AVAILABLE = False
    warn_if_broken("optuna", exc)


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
    horizon: Optional[int] = None,
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
    horizon : int or None
        Forecast horizon; each CV fold forecasts this many steps.

    Returns
    -------
    best_params : dict
        Best hyperparameters found (the full search-space dict, e.g. SARIMAX
        ``order``); empty when no trial produced a finite score.
    best_score : float
        Best metric score (lower is better); ``inf`` when no trial succeeded.
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna is required for tuning. Install with: pip install lazypredict[tune]"
        )

    from lazypredict.TimeSeriesForecasting import AutoARIMAForecaster

    space_fn = get_ts_search_space(model_name)
    if space_fn is None:
        logger.info("No search space for %s, skipping tuning.", model_name)
        return {}, float("inf")

    def objective(trial):
        params = space_fn(trial)
        if tune_seasonal and model_name in ("SARIMAX", "HoltWinters_Add", "HoltWinters_Mul"):
            candidates = _seasonal_candidates(
                len(y_train), seasonal_period,
                # A SARIMAX state space grows with the period; longer ones
                # need minutes and gigabytes per fit.
                max_period=AutoARIMAForecaster.MAX_SEASONAL_PERIOD if model_name == "SARIMAX" else None,
            )
            if candidates:
                params["seasonal_period"] = trial.suggest_categorical("seasonal_period", candidates)
        trial.set_user_attr("params", params)
        # MASE is scaled by the benchmark's seasonal period, not the trial's:
        # otherwise a period with a large naive error looks best.
        return _temporal_cv_error(
            wrapper, model_name, params, y_train, X_train,
            seasonal_period, tune_metric.lower(), cv, horizon=horizon,
        )

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

    finished = [
        (float(t.value), t)
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
    ]
    if not finished:
        return {}, float("inf")
    best_value, best = min(finished, key=lambda item: item[0])
    return dict(best.user_attrs.get("params", best.params)), best_value


def _seasonal_candidates(
    n_obs: int, seasonal_period: Optional[int], max_period: Optional[int] = None
) -> List[int]:
    """Seasonal periods worth trying for a series of *n_obs* values."""
    limit = min(n_obs // 3, max_period + 1) if max_period else n_obs // 3
    candidates = [p for p in CANDIDATE_SEASONAL_PERIODS if p < limit]
    if seasonal_period and seasonal_period < limit and seasonal_period not in candidates:
        candidates.append(seasonal_period)
    return sorted(candidates)


def _temporal_cv_error(
    wrapper, model_name, params, y_train, X_train, seasonal_period, metric_key, cv,
    horizon: Optional[int] = None,
) -> float:
    """Mean expanding-window CV error of *wrapper* configured with *params*.

    With *horizon*, every fold forecasts that many steps, as the benchmark
    does; otherwise the folds split the series evenly.
    """
    test_size = min(horizon, len(y_train) // (cv + 1)) if horizon else None
    scores = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=cv, test_size=test_size).split(y_train):
        y_cv_train = y_train[train_idx]
        X_cv_train = X_train[train_idx] if X_train is not None else None
        X_cv_val = X_train[val_idx] if X_train is not None else None
        try:
            w = copy.deepcopy(wrapper)
            _apply_params_to_wrapper(w, model_name, params)
            w.fit(y_cv_train, X_cv_train)
            y_pred = w.predict(len(val_idx), X_cv_val)
            fold_metrics = compute_forecast_metrics(
                y_train[val_idx], y_pred, y_cv_train, seasonal_period=seasonal_period or 1
            )
            scores.append(fold_metrics[metric_key])
        except Exception:
            scores.append(float("inf"))
    return float(np.mean(scores))


def _apply_params_to_wrapper(
    wrapper, model_name, params, n_lags=None, n_rolling_1=None, n_rolling_2=None, sp=None
):
    """Apply tuned parameters to a forecaster wrapper instance.

    *params* is a search-space dict: ``n_lags``/``n_rolling_1``/``n_rolling_2``
    configure the lag features of ML/DL forecasters, ``seasonal_period``
    the seasonal models, and everything else is stored in the wrapper's
    ``params`` (applied by its ``fit``/``predict``) or, for LSTM/GRU, set as
    attributes. Explicit keyword arguments override the dict entries.
    """
    from lazypredict.TimeSeriesForecasting import (
        HoltWintersForecaster,
        MLForecaster,
        _TorchRNNForecaster,
    )

    params = dict(params)
    feature_params = {
        key: params.pop(key, None)
        for key in ("n_lags", "n_rolling_1", "n_rolling_2", "seasonal_period")
    }
    n_lags = n_lags if n_lags is not None else feature_params["n_lags"]
    n_rolling_1 = n_rolling_1 if n_rolling_1 is not None else feature_params["n_rolling_1"]
    n_rolling_2 = n_rolling_2 if n_rolling_2 is not None else feature_params["n_rolling_2"]
    sp = sp if sp is not None else feature_params["seasonal_period"]

    if isinstance(wrapper, (MLForecaster, _TorchRNNForecaster)):
        if n_lags is not None:
            wrapper.n_lags = n_lags
        if n_rolling_1 is not None and n_rolling_2 is not None:
            wrapper.n_rolling = (n_rolling_1, n_rolling_2)
    if sp is not None:
        attribute = "seasonal_periods" if isinstance(wrapper, HoltWintersForecaster) else "seasonal_period"
        if hasattr(wrapper, attribute):
            setattr(wrapper, attribute, sp)
    if isinstance(wrapper, _TorchRNNForecaster):
        for key in ("hidden_size", "learning_rate", "batch_size", "n_epochs"):
            if key in params:
                setattr(wrapper, key, params.pop(key))
    wrapper.params = params


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
    refit: bool = True,
    horizon: Optional[int] = None,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """Tune the top-k forecasters from initial ranking.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Ranked model scores from initial benchmarking.
    models : dict
        Fitted model wrappers.
    all_wrappers : list of (name, wrapper) tuples
        All available forecaster wrappers.
    y_train : np.ndarray
        Training time series.
    X_train : np.ndarray or None
        Exogenous features.
    seasonal_period : int or None
        Seasonal period.
    top_k : int
        Number of top models to tune.
    tune_metric : str
        Metric to optimize.
    cv : int
        Number of CV folds.
    n_trials : int
        Optuna trials per model.
    timeout : float or None
        Seconds limit per model tuning.
    random_state : int
        Random seed.
    tune_seasonal : bool
        Search over seasonal_period values.
    refit : bool
        When True, refit each tuned model on full ``y_train`` with
        best parameters and return the fitted wrappers.

    Returns
    -------
    tune_results : pd.DataFrame
        One row per model (index ``Model``) with ``Best <tune_metric>`` (mean
        temporal-CV error of the best trial, lower is better), ``Best Params``
        (a dict) and ``Status`` (``"tuned"``, ``"no search space"`` or
        ``"failed: ..."``).
    tuned_models : dict
        ``{model_name: fitted_wrapper}`` when ``refit=True``,
        empty dict otherwise.
    """
    import pandas as pd

    top_model_names = list(scores_df.index[:top_k])
    name_to_wrapper = {name: w for name, w in all_wrappers}
    score_column = f"Best {tune_metric}"

    results = []
    tuned_models: Dict[str, Any] = {}

    for model_name in top_model_names:
        wrapper = name_to_wrapper.get(model_name)
        if wrapper is None:
            continue
        row = {"Model": model_name, score_column: float("nan"), "Best Params": {}}

        if get_ts_search_space(model_name) is None:
            logger.info("No search space for %s, skipping.", model_name)
            results.append({**row, "Status": "no search space"})
            continue

        logger.info("Tuning forecaster %s...", model_name)
        try:
            best_params, best_score = tune_forecaster_optuna(
                model_name, wrapper, y_train, X_train, seasonal_period,
                tune_metric=tune_metric, cv=cv, n_trials=n_trials,
                timeout=timeout, random_state=random_state,
                tune_seasonal=tune_seasonal, horizon=horizon,
            )
        except Exception as exc:
            logger.warning("Tuning %s failed: %s", model_name, exc)
            results.append({**row, "Status": f"failed: {exc}"})
            continue
        if not best_params:
            results.append({**row, "Status": "failed: no successful trial"})
            continue
        results.append({
            **row, score_column: best_score, "Best Params": best_params, "Status": "tuned",
        })

        # Refit with best params on full training data
        if refit:
            try:
                w = copy.deepcopy(wrapper)
                _apply_params_to_wrapper(w, model_name, best_params)
                w.fit(y_train, X_train)
                tuned_models[model_name] = w
                logger.info("Refit %s with best params.", model_name)
            except Exception as exc:
                logger.warning(
                    "Failed to refit %s with best params: %s", model_name, exc
                )

    columns = ["Model", score_column, "Best Params", "Status"]
    return pd.DataFrame(results, columns=columns).set_index("Model"), tuned_models
