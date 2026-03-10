"""Hyperparameter search space definitions for time series forecasting models.

Each function takes an Optuna ``trial`` and returns a dict of hyperparameters.
ML/DL forecasters also expose feature-engineering params (n_lags, n_rolling)
as tunable hyperparameters — unique to the forecasting case.
"""

from typing import Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------


def _sarimax_space(trial) -> dict:
    p = trial.suggest_int("p", 0, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)
    return {"order": (p, d, q)}


def _exp_smoothing_space(trial) -> dict:
    return {
        "smoothing_level": trial.suggest_float("smoothing_level", 0.01, 0.99),
        "smoothing_trend": trial.suggest_float("smoothing_trend", 0.01, 0.99),
        "damped_trend": trial.suggest_categorical("damped_trend", [True, False]),
    }


def _holtwinters_space(trial) -> dict:
    return {
        "smoothing_level": trial.suggest_float("smoothing_level", 0.01, 0.99),
        "smoothing_trend": trial.suggest_float("smoothing_trend", 0.01, 0.99),
        "smoothing_seasonal": trial.suggest_float("smoothing_seasonal", 0.01, 0.99),
        "damped_trend": trial.suggest_categorical("damped_trend", [True, False]),
    }


def _theta_space(trial) -> dict:
    return {
        "theta": trial.suggest_float("theta", 0.5, 5.0),
    }


def _auto_arima_space(trial) -> dict:
    return {
        "max_p": trial.suggest_int("max_p", 1, 5),
        "max_q": trial.suggest_int("max_q", 1, 5),
        "max_d": trial.suggest_int("max_d", 1, 2),
        "seasonal": trial.suggest_categorical("seasonal", [True, False]),
        "stepwise": trial.suggest_categorical("stepwise", [True, False]),
        "information_criterion": trial.suggest_categorical(
            "information_criterion", ["aic", "bic"]
        ),
    }


# ---------------------------------------------------------------------------
# ML models (feature-engineering params + model hyperparams)
# ---------------------------------------------------------------------------


def _ml_base_space(trial) -> dict:
    """Feature-engineering params shared by all ML forecasters."""
    return {
        "n_lags": trial.suggest_int("n_lags", 3, 30),
        "n_rolling_1": trial.suggest_int("n_rolling_1", 2, 14),
        "n_rolling_2": trial.suggest_int("n_rolling_2", 5, 30),
    }


def _linear_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    return base


def _ridge_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["alpha"] = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    return base


def _lasso_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["alpha"] = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    return base


def _elasticnet_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["alpha"] = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    base["l1_ratio"] = trial.suggest_float("l1_ratio", 0.01, 0.99)
    return base


def _knn_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 30)
    base["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
    return base


def _dt_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["max_depth"] = trial.suggest_int("max_depth", 2, 30)
    base["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
    return base


def _rf_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
    base["max_depth"] = trial.suggest_int("max_depth", 2, 30)
    base["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 20)
    return base


def _gbr_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
    base["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    base["max_depth"] = trial.suggest_int("max_depth", 2, 15)
    base["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    return base


def _adaboost_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
    base["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)
    return base


def _extra_trees_ts_space(trial) -> dict:
    return _rf_ts_space(trial)


def _bagging_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_estimators"] = trial.suggest_int("n_estimators", 10, 200)
    base["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
    return base


def _svr_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["C"] = trial.suggest_float("C", 1e-3, 100.0, log=True)
    base["kernel"] = trial.suggest_categorical("kernel", ["rbf", "linear"])
    return base


def _xgb_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
    base["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    base["max_depth"] = trial.suggest_int("max_depth", 2, 15)
    base["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    base["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    base["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    base["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    return base


def _lgbm_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
    base["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    base["max_depth"] = trial.suggest_int("max_depth", 2, 15)
    base["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
    base["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    base["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    base["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    return base


def _catboost_ts_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["iterations"] = trial.suggest_int("iterations", 50, 300)
    base["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    base["depth"] = trial.suggest_int("depth", 2, 10)
    base["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True)
    return base


# ---------------------------------------------------------------------------
# Deep Learning models
# ---------------------------------------------------------------------------


def _lstm_space(trial) -> dict:
    base = _ml_base_space(trial)
    base["hidden_size"] = trial.suggest_int("hidden_size", 16, 128)
    base["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    base["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
    base["n_epochs"] = trial.suggest_int("n_epochs", 20, 100)
    return base


def _gru_space(trial) -> dict:
    return _lstm_space(trial)


# ---------------------------------------------------------------------------
# Registry: model name -> search space function
# ---------------------------------------------------------------------------

TS_SEARCH_SPACES: Dict[str, Callable] = {
    # Statistical
    "SARIMAX": _sarimax_space,
    "Holt": _exp_smoothing_space,
    "HoltWinters_Add": _holtwinters_space,
    "HoltWinters_Mul": _holtwinters_space,
    "Theta": _theta_space,
    "AutoARIMA": _auto_arima_space,
    # ML
    "LinearRegression_TS": _linear_ts_space,
    "Ridge_TS": _ridge_ts_space,
    "Lasso_TS": _lasso_ts_space,
    "ElasticNet_TS": _elasticnet_ts_space,
    "KNeighborsRegressor_TS": _knn_ts_space,
    "DecisionTreeRegressor_TS": _dt_ts_space,
    "RandomForestRegressor_TS": _rf_ts_space,
    "GradientBoostingRegressor_TS": _gbr_ts_space,
    "AdaBoostRegressor_TS": _adaboost_ts_space,
    "ExtraTreesRegressor_TS": _extra_trees_ts_space,
    "BaggingRegressor_TS": _bagging_ts_space,
    "SVR_TS": _svr_ts_space,
    "XGBRegressor_TS": _xgb_ts_space,
    "LGBMRegressor_TS": _lgbm_ts_space,
    "CatBoostRegressor_TS": _catboost_ts_space,
    # DL
    "LSTM_TS": _lstm_space,
    "GRU_TS": _gru_space,
}


def get_ts_search_space(model_name: str) -> Optional[Callable]:
    """Return the search space function for a forecaster, or None."""
    return TS_SEARCH_SPACES.get(model_name)
