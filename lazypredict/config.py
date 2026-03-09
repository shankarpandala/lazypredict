"""Configuration constants and defaults for LazyPredict."""

from typing import FrozenSet, Tuple

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
