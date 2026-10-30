"""Hyperparameter tuning engine for LazyPredict.

Provides Optuna-based and sklearn-based tuning backends for both
supervised models and time series forecasters.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from lazypredict.search_spaces import SEARCH_SPACES, get_search_space
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

# Optional FLAML
try:
    import flaml.tune  # noqa: F401

    _FLAML_AVAILABLE = True
except Exception as exc:  # missing, or installed but broken
    _FLAML_AVAILABLE = False
    warn_if_broken("flaml", exc)


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

# Keyword arguments that only control logging/parallelism, not the model.
_NON_MODEL_PARAMS = ("n_jobs", "verbose", "verbosity", "random_state", "seed", "thread_count")


def _model_kwargs(model_class: Any, random_state: int, use_gpu: bool = False) -> Dict[str, Any]:
    """Seed, GPU and verbosity keyword arguments shared by every trial."""
    from lazypredict.config import get_gpu_model_params

    kwargs = dict(get_gpu_model_params(model_class, use_gpu))
    module = getattr(model_class, "__module__", "") or ""
    if "catboost" in module:
        kwargs.setdefault("verbose", 0)
    if "lightgbm" in module:
        kwargs.setdefault("verbose", -1)
        kwargs.setdefault("verbosity", -1)
    if "xgboost" in module:
        kwargs.setdefault("verbosity", 0)
    try:
        if "random_state" in model_class().get_params():
            kwargs["random_state"] = random_state
    except Exception:
        pass
    return kwargs


def _cv_score(pipe: Any, X_train: Any, y_train: Any, cv: int, scoring: str, n_jobs: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs, error_score=np.nan
        )
    return float(np.mean(scores))


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

    Trials that raise or produce NaN scores are recorded as failed and the
    search continues. The sampler is seeded with *random_state*.

    Returns
    -------
    best_params : dict
        Best hyperparameters found (including fixed values from the search
        space); empty when no trial completed.
    best_score : float
        Best mean cross-validation score (higher is better); NaN when no
        trial completed.
    """
    _check_optuna()
    from sklearn.pipeline import Pipeline

    space_fn = get_search_space(model_name)
    if space_fn is None:
        logger.info("No search space for %s, skipping tuning.", model_name)
        return {}, float("nan")
    fixed = _model_kwargs(model_class, random_state, use_gpu)

    def objective(trial):
        params = space_fn(trial)
        trial.set_user_attr("params", params)
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", model_class(**{**fixed, **params}))])
        return _cv_score(pipe, X_train, y_train, cv, scoring, n_jobs)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {}, float("nan")
    best = study.best_trial
    return dict(best.user_attrs.get("params", best.params)), float(study.best_value)


class _DistributionRecorder:
    """Stand-in for an Optuna trial that turns a search space into scipy distributions."""

    def suggest_float(self, name, low, high, log=False, **kwargs):
        from scipy import stats

        return stats.loguniform(low, high) if log else stats.uniform(low, high - low)

    def suggest_int(self, name, low, high, **kwargs):
        from scipy import stats

        return stats.randint(low, high + 1)

    def suggest_categorical(self, name, choices):
        return list(choices)


def _optuna_to_sklearn_dist(model_name: str) -> dict:
    """Convert a model's Optuna search space into RandomizedSearchCV distributions."""
    space_fn = get_search_space(model_name)
    if space_fn is None:
        return {}
    distributions = {}
    for key, value in space_fn(_DistributionRecorder()).items():
        if isinstance(value, list) or hasattr(value, "rvs"):
            distributions[key] = value
        else:  # a fixed value in the search space
            distributions[key] = [value]
    return distributions


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
    use_gpu: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Tune a single supervised model using scikit-learn's RandomizedSearchCV.

    Distributions are derived from the model's Optuna search space.
    Candidates that fail (e.g. ``n_neighbors`` larger than a fold) score NaN
    instead of aborting the search.
    """
    from sklearn.pipeline import Pipeline

    param_dist = _optuna_to_sklearn_dist(model_name)
    if not param_dist:
        return {}, float("nan")

    model = model_class(**_model_kwargs(model_class, random_state, use_gpu))
    pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])
    prefixed = {f"model__{k}": v for k, v in param_dist.items()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            pipe,
            prefixed,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)

    best = {k[len("model__"):]: v for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


# scikit-learn / boosting class name -> FLAML learner name
_FLAML_LEARNERS = {
    "RandomForestClassifier": "rf",
    "RandomForestRegressor": "rf",
    "ExtraTreesClassifier": "extra_tree",
    "ExtraTreesRegressor": "extra_tree",
    "KNeighborsClassifier": "kneighbor",
    "KNeighborsRegressor": "kneighbor",
    "LogisticRegression": "lrl2",
    "LGBMClassifier": "lgbm",
    "LGBMRegressor": "lgbm",
    "XGBClassifier": "xgboost",
    "XGBRegressor": "xgboost",
    "CatBoostClassifier": "catboost",
    "CatBoostRegressor": "catboost",
}


def _flaml_loss(scoring: str):
    """FLAML minimises a loss; wrap a scikit-learn scorer as ``1 - score``."""
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)

    def loss(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
        score = scorer(estimator, X_val, y_val)
        return 1.0 - score, {scoring: score}

    return loss


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
    use_gpu: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Tune a single supervised model with FLAML's cost-frugal search.

    FLAML searches its own space for the matching learner; the parameters of
    the best estimator it finds are then scored through the lazypredict
    pipeline, so ``best_score`` is comparable with the other backends.
    Models FLAML has no learner for return ``({}, nan)``.
    """
    _check_flaml()
    from flaml import AutoML
    from sklearn.base import ClassifierMixin
    from sklearn.pipeline import Pipeline

    learner = _FLAML_LEARNERS.get(model_name)
    if learner is None:
        logger.info("FLAML has no learner for %s, skipping tuning.", model_name)
        return {}, float("nan")

    automl = AutoML()
    settings = {
        "task": "classification" if issubclass(model_class, ClassifierMixin) else "regression",
        "metric": _flaml_loss(scoring),
        "estimator_list": [learner],
        "max_iter": n_trials,
        "time_budget": timeout if timeout is not None else 60,
        "eval_method": "cv",
        "n_splits": cv,
        "n_jobs": n_jobs,
        "seed": random_state,
        "verbose": 0,
    }
    X_prepared = preprocessor.fit_transform(X_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        automl.fit(X_prepared, np.asarray(y_train), **settings)

    estimator = getattr(automl.model, "estimator", None)
    if estimator is None or type(estimator).__name__ != model_class.__name__:
        return {}, float("nan")
    defaults = model_class().get_params()
    best = {
        key: value
        for key, value in estimator.get_params().items()
        if key in defaults and key not in _NON_MODEL_PARAMS and defaults[key] != value
    }
    fixed = _model_kwargs(model_class, random_state, use_gpu)
    pipe = Pipeline([("preprocessor", clone(preprocessor)), ("model", model_class(**{**fixed, **best}))])
    return best, _cv_score(pipe, X_train, y_train, cv, scoring, n_jobs)


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
    cv: int = 5,
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
        Preprocessor template (cloned for every candidate).
    scoring : str
        Scoring metric string (e.g. 'balanced_accuracy', 'r2').
    top_k : int
        Number of top models to tune.
    n_trials : int
        Trials (Optuna/FLAML) or sampled candidates (sklearn) per model.
    timeout : float or None
        Seconds per model tuning (Optuna and FLAML).
    random_state : int
        Random seed.
    n_jobs : int
        Parallel jobs.
    backend : str
        'optuna', 'sklearn', or 'flaml'.
    use_gpu : bool
        Whether GPU is enabled.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        One row per model (index ``Model``) with ``Best Score`` (mean
        cross-validated score on the training data, higher is better),
        ``Best Params`` (a dict) and ``Status`` (``"tuned"``,
        ``"no search space"`` or ``"failed: ..."``). A model that cannot be
        tuned never aborts the others.
    """
    tuners = {
        "optuna": lambda name, cls: tune_supervised_optuna(
            name, cls, X_train, y_train, preprocessor, scoring, cv=cv, n_trials=n_trials,
            timeout=timeout, random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
        ),
        "sklearn": lambda name, cls: tune_supervised_sklearn(
            name, cls, X_train, y_train, preprocessor, scoring, cv=cv, n_iter=n_trials,
            random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
        ),
        "flaml": lambda name, cls: tune_supervised_flaml(
            name, cls, X_train, y_train, preprocessor, scoring, cv=cv, n_trials=n_trials,
            timeout=timeout, random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
        ),
    }
    if backend not in tuners:
        raise ValueError(f"Unknown backend: {backend}. Use 'optuna', 'sklearn', or 'flaml'.")
    supported = _FLAML_LEARNERS if backend == "flaml" else SEARCH_SPACES

    name_to_class = dict(estimator_list)
    results = []
    for model_name in list(scores_df.index[:top_k]):
        model_class = name_to_class.get(model_name)
        if model_class is None:
            continue
        row = {"Model": model_name, "Best Score": float("nan"), "Best Params": {}}
        if model_name not in supported:
            logger.info("No %s search space for %s, skipping.", backend, model_name)
            results.append({**row, "Status": "no search space"})
            continue

        logger.info("Tuning %s with %s backend...", model_name, backend)
        try:
            best_params, best_score = tuners[backend](model_name, model_class)
        except Exception as exc:
            logger.warning("Tuning %s failed: %s", model_name, exc)
            results.append({**row, "Status": f"failed: {exc}"})
            continue
        if not best_params or np.isnan(best_score):
            results.append({**row, "Status": "failed: no successful trial"})
            continue
        results.append({**row, "Best Score": best_score, "Best Params": best_params, "Status": "tuned"})

    columns = ["Model", "Best Score", "Best Params", "Status"]
    return pd.DataFrame(results, columns=columns).set_index("Model")
