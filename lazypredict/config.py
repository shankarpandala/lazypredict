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

    return {}
