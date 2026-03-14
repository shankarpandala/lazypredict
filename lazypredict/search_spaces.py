"""Hyperparameter search space definitions for supervised models.

Each function takes an Optuna ``trial`` and returns a dict of hyperparameters
to pass to the model constructor.  Models not listed here are skipped during
tuning (they have no meaningful hyperparameters or are too niche).
"""

from typing import Callable, Dict, Optional


def _ridge_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}


def _lasso_space(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}


def _elasticnet_space(trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
    }


def _sgd_space(trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
    }


def _knn_space(trial) -> dict:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
    }


def _decision_tree_space(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _random_forest_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _extra_trees_space(trial) -> dict:
    return _random_forest_space(trial)


def _gradient_boosting_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _adaboost_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
    }


def _bagging_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }


def _svc_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _svr_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
    }


def _xgb_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 15),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _catboost_space(trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
    }


def _passive_aggressive_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
    }


def _huber_space(trial) -> dict:
    return {
        "epsilon": trial.suggest_float("epsilon", 1.0, 5.0),
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
    }


def _nu_svc_space(trial) -> dict:
    return {
        "nu": trial.suggest_float("nu", 0.01, 0.99),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _nu_svr_space(trial) -> dict:
    return {
        "nu": trial.suggest_float("nu", 0.01, 0.99),
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


# ---------------------------------------------------------------------------
# Registry: model class name -> search space function
# ---------------------------------------------------------------------------

SEARCH_SPACES: Dict[str, Callable] = {
    # Linear
    "Ridge": _ridge_space,
    "RidgeClassifier": _ridge_space,
    "Lasso": _lasso_space,
    "LassoCV": _lasso_space,
    "ElasticNet": _elasticnet_space,
    "ElasticNetCV": _elasticnet_space,
    "SGDClassifier": _sgd_space,
    "SGDRegressor": _sgd_space,
    "PassiveAggressiveClassifier": _passive_aggressive_space,
    "PassiveAggressiveRegressor": _passive_aggressive_space,
    "HuberRegressor": _huber_space,
    # KNN
    "KNeighborsClassifier": _knn_space,
    "KNeighborsRegressor": _knn_space,
    # Trees
    "DecisionTreeClassifier": _decision_tree_space,
    "DecisionTreeRegressor": _decision_tree_space,
    "RandomForestClassifier": _random_forest_space,
    "RandomForestRegressor": _random_forest_space,
    "ExtraTreesClassifier": _extra_trees_space,
    "ExtraTreesRegressor": _extra_trees_space,
    "GradientBoostingClassifier": _gradient_boosting_space,
    "GradientBoostingRegressor": _gradient_boosting_space,
    "AdaBoostClassifier": _adaboost_space,
    "AdaBoostRegressor": _adaboost_space,
    "BaggingClassifier": _bagging_space,
    "BaggingRegressor": _bagging_space,
    # SVM
    "SVC": _svc_space,
    "SVR": _svr_space,
    "NuSVC": _nu_svc_space,
    "NuSVR": _nu_svr_space,
    # Boosting (optional)
    "XGBClassifier": _xgb_space,
    "XGBRegressor": _xgb_space,
    "LGBMClassifier": _lgbm_space,
    "LGBMRegressor": _lgbm_space,
    "CatBoostClassifier": _catboost_space,
    "CatBoostRegressor": _catboost_space,
}


def get_search_space(model_name: str) -> Optional[Callable]:
    """Return the search space function for a model, or None if not available."""
    return SEARCH_SPACES.get(model_name)
