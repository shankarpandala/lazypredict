"""Hyperparameter tuning engine for LazyPredict.

Provides Optuna-based and sklearn-based tuning backends for both
supervised models and time series forecasters.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from lazypredict.search_spaces import get_search_space

logger = logging.getLogger("lazypredict")

# Optional Optuna
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

# Optional FLAML
try:
    import flaml.tune  # noqa: F401

    _FLAML_AVAILABLE = True
except ImportError:
    _FLAML_AVAILABLE = False


def _check_optuna():
    if not _OPTUNA_AVAILABLE:
        raise ImportError(
            "optuna is required for hyperparameter tuning. "
            "Install with: pip install lazypredict[tune]"
        )


def _check_flaml():
    if not _FLAML_AVAILABLE:
        raise ImportError(
            "flaml is required for FLAML tuning backend. "
            "Install with: pip install lazypredict[flaml]"
        )


# ---------------------------------------------------------------------------
# Supervised tuning
# ---------------------------------------------------------------------------


def tune_supervised_optuna(
    model_name: str,
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: Any,
    preprocessor: Any,
    scoring: str,
    cv: int = 5,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    use_gpu: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Tune a single supervised model using Optuna.

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    best_score : float
        Best cross-validation score.
    """
    _check_optuna()
    from sklearn.pipeline import Pipeline

    from lazypredict.config import get_gpu_model_params

    space_fn = get_search_space(model_name)
    if space_fn is None:
        logger.info("No search space for %s, skipping tuning.", model_name)
        return {}, float("-inf")

    def objective(trial):
        params = space_fn(trial)
        gpu_params = get_gpu_model_params(model_class, use_gpu)
        all_params = {**gpu_params, **params}
        # Suppress boosting verbose
        module = getattr(model_class, "__module__", "") or ""
        if "catboost" in module:
            all_params.setdefault("verbose", 0)
        if "lightgbm" in module:
            all_params.setdefault("verbose", -1)
            all_params.setdefault("verbosity", -1)
        if "xgboost" in module:
            all_params.setdefault("verbosity", 0)
        try:
            if "random_state" in model_class().get_params():
                all_params["random_state"] = random_state
        except Exception:
            pass

        model = model_class(**all_params)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study.best_params, study.best_value


def tune_supervised_sklearn(
    model_name: str,
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: Any,
    preprocessor: Any,
    scoring: str,
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], float]:
    """Tune a single supervised model using sklearn RandomizedSearchCV.

    Uses a simple parameter distribution derived from the search space registry.
    """
    from sklearn.pipeline import Pipeline

    space_fn = get_search_space(model_name)
    if space_fn is None:
        return {}, float("-inf")

    # Build a rough param distribution from the Optuna space
    param_dist = _optuna_to_sklearn_dist(model_name)
    if not param_dist:
        return {}, float("-inf")

    model = model_class()
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Prefix params with step name
    prefixed = {f"model__{k}": v for k, v in param_dist.items()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            pipe,
            prefixed,
            n_iter=min(n_iter, 50),
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            error_score="raise",
        )
        search.fit(X_train, y_train)

    # Strip prefix
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return best, search.best_score_


def _optuna_to_sklearn_dist(model_name: str) -> dict:
    """Convert Optuna search space to rough scipy distributions for RandomizedSearchCV."""
    from scipy import stats

    distributions: Dict[str, Dict[str, Any]] = {
        "Ridge": {"alpha": stats.loguniform(1e-4, 100)},
        "RidgeClassifier": {"alpha": stats.loguniform(1e-4, 100)},
        "Lasso": {"alpha": stats.loguniform(1e-4, 100)},
        "ElasticNet": {
            "alpha": stats.loguniform(1e-4, 100),
            "l1_ratio": stats.uniform(0.01, 0.98),
        },
        "KNeighborsClassifier": {
            "n_neighbors": stats.randint(1, 51),
            "weights": ["uniform", "distance"],
        },
        "KNeighborsRegressor": {
            "n_neighbors": stats.randint(1, 51),
            "weights": ["uniform", "distance"],
        },
        "RandomForestClassifier": {
            "n_estimators": stats.randint(50, 501),
            "max_depth": stats.randint(2, 51),
        },
        "RandomForestRegressor": {
            "n_estimators": stats.randint(50, 501),
            "max_depth": stats.randint(2, 51),
        },
        "DecisionTreeClassifier": {"max_depth": stats.randint(2, 51)},
        "DecisionTreeRegressor": {"max_depth": stats.randint(2, 51)},
        "SVC": {"C": stats.loguniform(1e-3, 100), "kernel": ["rbf", "linear"]},
        "SVR": {"C": stats.loguniform(1e-3, 100), "kernel": ["rbf", "linear"]},
    }
    return distributions.get(model_name, {})


def tune_supervised_flaml(
    model_name: str,
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: Any,
    preprocessor: Any,
    scoring: str,
    cv: int = 5,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], float]:
    """Tune using FLAML's cost-frugal search."""
    _check_flaml()
    # FLAML tune requires a config space and evaluation function
    # Wrap it to use the same interface as Optuna
    # For now, fall back to Optuna-like behavior via FLAML's BlendSearch
    logger.info("FLAML tuning for %s", model_name)
    # Use FLAML's AutoML for individual model tuning
    from flaml import AutoML

    automl = AutoML()
    settings = {
        "time_budget": timeout or 60,
        "metric": scoring.replace("neg_", "").replace("_", " "),
        "task": "classification" if "Classifier" in model_name else "regression",
        "estimator_list": [model_name.lower()],
        "seed": random_state,
        "verbose": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            automl.fit(X_train, y_train, **settings)
            return automl.best_config or {}, automl.best_loss or float("-inf")
        except Exception as exc:
            logger.warning("FLAML tuning failed for %s: %s", model_name, exc)
            return {}, float("-inf")


# ---------------------------------------------------------------------------
# Top-K tuning orchestrator
# ---------------------------------------------------------------------------


def tune_top_k(
    scores_df: pd.DataFrame,
    models: Dict[str, Any],
    estimator_list: List[Tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: Any,
    preprocessor: Any,
    scoring: str,
    top_k: int = 5,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    backend: str = "optuna",
    use_gpu: bool = False,
) -> pd.DataFrame:
    """Tune the top-k models from the initial ranking.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Scores DataFrame from initial fit (index = model names).
    models : dict
        Fitted model pipelines.
    estimator_list : list
        List of (name, class) tuples from _get_estimator_list.
    X_train, y_train : array-like
        Training data.
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    scoring : str
        Scoring metric string (e.g. 'balanced_accuracy', 'r2').
    top_k : int
        Number of top models to tune.
    n_trials : int
        Optuna trials per model.
    timeout : float or None
        Seconds per model tuning.
    random_state : int
        Random seed.
    n_jobs : int
        Parallel jobs.
    backend : str
        'optuna', 'sklearn', or 'flaml'.
    use_gpu : bool
        Whether GPU is enabled.

    Returns
    -------
    pd.DataFrame
        Tuning results with columns: Model, Best Score, Best Params.
    """
    top_model_names = list(scores_df.index[:top_k])
    name_to_class = {name: cls for name, cls in estimator_list}

    results = []
    for model_name in top_model_names:
        model_class = name_to_class.get(model_name)
        if model_class is None:
            continue

        space_fn = get_search_space(model_name)
        if space_fn is None:
            logger.info("No search space for %s, skipping.", model_name)
            results.append({
                "Model": model_name,
                "Best Score": None,
                "Best Params": "N/A (no search space)",
            })
            continue

        logger.info("Tuning %s with %s backend...", model_name, backend)

        if backend == "optuna":
            best_params, best_score = tune_supervised_optuna(
                model_name, model_class, X_train, y_train, preprocessor,
                scoring, cv=5, n_trials=n_trials, timeout=timeout,
                random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
            )
        elif backend == "sklearn":
            best_params, best_score = tune_supervised_sklearn(
                model_name, model_class, X_train, y_train, preprocessor,
                scoring, cv=5, n_iter=n_trials, random_state=random_state,
                n_jobs=n_jobs,
            )
        elif backend == "flaml":
            best_params, best_score = tune_supervised_flaml(
                model_name, model_class, X_train, y_train, preprocessor,
                scoring, cv=5, n_trials=n_trials, timeout=timeout,
                random_state=random_state, n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'optuna', 'sklearn', or 'flaml'.")

        results.append({
            "Model": model_name,
            "Best Score": best_score,
            "Best Params": str(best_params),
        })

    return pd.DataFrame(results).set_index("Model")
