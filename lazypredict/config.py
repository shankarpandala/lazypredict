"""Configuration constants and defaults for LazyPredict."""

import logging
from typing import FrozenSet, Tuple

logger = logging.getLogger("lazypredict")

# ---------------------------------------------------------------------------
# Valid encoder types
# ---------------------------------------------------------------------------
VALID_ENCODERS: Tuple[str, ...] = ("onehot", "ordinal", "target", "binary")

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_CARDINALITY_THRESHOLD: int = 11
DEFAULT_IMPUTE_STRATEGY: str = "mean"
DEFAULT_CATEGORICAL_FILL: str = "missing"
DEFAULT_UNKNOWN_VALUE: int = -1
DEFAULT_N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Removed estimators — excluded from default model lists
# ---------------------------------------------------------------------------
REMOVED_CLASSIFIERS: FrozenSet[str] = frozenset([
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV",
    "MultiOutputClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
])

REMOVED_REGRESSORS: FrozenSet[str] = frozenset([
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "IsotonicRegression",
    "StackingRegressor",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "VotingRegressor",
])

# ---------------------------------------------------------------------------
# Time series forecasting defaults
# ---------------------------------------------------------------------------
DEFAULT_N_LAGS: int = 10
DEFAULT_ROLLING_WINDOWS: Tuple[int, ...] = (3, 7)
DEFAULT_SORT_BY_FORECASTER: str = "RMSE"
REMOVED_FORECASTERS: FrozenSet[str] = frozenset()


# ---------------------------------------------------------------------------
# GPU support utilities
# ---------------------------------------------------------------------------
def is_gpu_available() -> bool:
    """Check if a CUDA-capable GPU is available via PyTorch.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_model_params(model_class, use_gpu: bool) -> dict:
    """Return GPU-related keyword arguments for a model class.

    Inspects the model class module to determine if it supports GPU
    acceleration and returns the appropriate kwargs.

    Supported GPU backends:

    - **XGBoost**: ``device="cuda"``
    - **LightGBM**: ``device="gpu"``
    - **CatBoost**: ``task_type="GPU"``
    - **cuML (RAPIDS)**: No extra params needed (GPU-native).

    Parameters
    ----------
    model_class : type
        The model class to inspect.
    use_gpu : bool
        Whether GPU usage has been requested by the user.

    Returns
    -------
    dict
        Keyword arguments to pass to the model constructor for GPU support.
        Empty dict if the model does not support GPU or ``use_gpu`` is False.
    """
    if not use_gpu:
        return {}

    module = getattr(model_class, "__module__", "") or ""

    # XGBoost: use device="cuda"
    if "xgboost" in module:
        if is_gpu_available():
            return {"device": "cuda"}
        else:
            logger.warning(
                "GPU requested for %s but CUDA is not available. "
                "Falling back to CPU.",
                model_class.__name__,
            )
            return {}

    # LightGBM: use device="gpu"
    if "lightgbm" in module:
        if is_gpu_available():
            return {"device": "gpu"}
        else:
            logger.warning(
                "GPU requested for %s but CUDA is not available. "
                "Falling back to CPU.",
                model_class.__name__,
            )
            return {}

    # CatBoost: use task_type="GPU"
    if "catboost" in module:
        if is_gpu_available():
            return {"task_type": "GPU"}
        else:
            logger.warning(
                "GPU requested for %s but CUDA is not available. "
                "Falling back to CPU.",
                model_class.__name__,
            )
            return {}

    # cuML (RAPIDS): models are GPU-native, no extra params needed
    if "cuml" in module:
        if is_gpu_available():
            return {}
        else:
            logger.warning(
                "GPU requested for %s (cuML) but CUDA is not available. "
                "cuML requires a CUDA-capable GPU.",
                model_class.__name__,
            )
            return {}

    return {}


def get_cuml_models() -> dict:
    """Return a mapping of sklearn model names to cuML GPU equivalents.

    cuML (RAPIDS) provides GPU-accelerated drop-in replacements for many
    scikit-learn estimators.  This function returns the available ones.

    Returns
    -------
    dict
        ``{sklearn_name: cuml_class}`` for available cuML models.
        Empty dict if cuML is not installed.
    """
    models: dict = {}
    try:
        import cuml  # noqa: F401
    except ImportError:
        return models

    # Classifiers
    _cuml_classifiers = [
        ("cuML_LogisticRegression", "cuml.linear_model", "LogisticRegression"),
        ("cuML_RandomForestClassifier", "cuml.ensemble", "RandomForestClassifier"),
        ("cuML_KNeighborsClassifier", "cuml.neighbors", "KNeighborsClassifier"),
        ("cuML_SVC", "cuml.svm", "SVC"),
    ]
    # Regressors
    _cuml_regressors = [
        ("cuML_LinearRegression", "cuml.linear_model", "LinearRegression"),
        ("cuML_Ridge", "cuml.linear_model", "Ridge"),
        ("cuML_Lasso", "cuml.linear_model", "Lasso"),
        ("cuML_ElasticNet", "cuml.linear_model", "ElasticNet"),
        ("cuML_RandomForestRegressor", "cuml.ensemble", "RandomForestRegressor"),
        ("cuML_KNeighborsRegressor", "cuml.neighbors", "KNeighborsRegressor"),
        ("cuML_SVR", "cuml.svm", "SVR"),
    ]

    for name, mod_path, cls_name in _cuml_classifiers + _cuml_regressors:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            models[name] = getattr(mod, cls_name)
        except (ImportError, AttributeError):
            pass

    return models
