"""
Supervised Models — LazyClassifier and LazyRegressor for rapid model benchmarking.

Provides LazyClassifier and LazyRegressor classes that train multiple
scikit-learn models with minimal code to quickly identify which algorithms
perform best on a given dataset.
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import logging
import os
import signal
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import all_estimators
from tqdm import tqdm

# Module-level logger — users can configure via logging.getLogger("lazypredict")
logger = logging.getLogger("lazypredict")

# Detect Jupyter notebook environment for tqdm
try:
    from IPython import get_ipython
    if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
        from tqdm.notebook import tqdm as notebook_tqdm
        _use_notebook_tqdm = True
    else:
        _use_notebook_tqdm = False
except Exception:
    _use_notebook_tqdm = False

# Optional category_encoders import with fallback
try:
    from category_encoders import BinaryEncoder, TargetEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

# Optional xgboost
try:
    import xgboost
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# Optional lightgbm
try:
    import lightgbm
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False

# Optional perpetual
try:
    from perpetual import PerpetualBooster
    PERPETUAL_AVAILABLE = True
except ImportError:
    PERPETUAL_AVAILABLE = False

# Intel Extension for Scikit-learn for better performance
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    INTEL_EXTENSION_AVAILABLE = True
except ImportError:
    INTEL_EXTENSION_AVAILABLE = False

# Optional MLflow for model tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Valid encoder types
# ---------------------------------------------------------------------------
_VALID_ENCODERS = ("onehot", "ordinal", "target", "binary")

# ---------------------------------------------------------------------------
# Removed estimators — these are excluded from the default model lists
# ---------------------------------------------------------------------------
_REMOVED_CLASSIFIERS = frozenset([
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

_REMOVED_REGRESSORS = frozenset([
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

# Kept as module-level lists for backward compatibility but built fresh
CLASSIFIERS: List[Tuple[str, Any]] = [
    est
    for est in all_estimators()
    if issubclass(est[1], ClassifierMixin) and est[0] not in _REMOVED_CLASSIFIERS
]

REGRESSORS: List[Tuple[str, Any]] = [
    est
    for est in all_estimators()
    if issubclass(est[1], RegressorMixin) and est[0] not in _REMOVED_REGRESSORS
]

# Append optional boosting models
if _XGBOOST_AVAILABLE:
    REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
    CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))

if _LIGHTGBM_AVAILABLE:
    REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
    CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))

if PERPETUAL_AVAILABLE:
    REGRESSORS.append(("PerpetualBooster", PerpetualBooster))
    CLASSIFIERS.append(("PerpetualBooster", PerpetualBooster))

# Backward-compatible aliases for removed_ lists
removed_classifiers = list(_REMOVED_CLASSIFIERS)
removed_regressors = list(_REMOVED_REGRESSORS)

# ---------------------------------------------------------------------------
# Default preprocessing pipelines
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder()),
    ]
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_mlflow_tracking_enabled() -> bool:
    """Check if MLflow tracking is enabled via the MLFLOW_TRACKING_URI environment variable."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return MLFLOW_AVAILABLE and tracking_uri is not None


def setup_mlflow() -> bool:
    """Initialize MLflow if tracking URI is set through environment variable.

    Returns
    -------
    bool
        True if MLflow was successfully configured, False otherwise.
    """
    if is_mlflow_tracking_enabled():
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.autolog()
        return True
    return False


def get_categorical_encoder(
    encoder_type: str = "onehot", cardinality: str = "low"
) -> Pipeline:
    """Get categorical encoder pipeline based on encoder type and cardinality.

    Parameters
    ----------
    encoder_type : str, optional (default='onehot')
        Type of encoder: 'onehot', 'ordinal', 'target', or 'binary'.
    cardinality : str, optional (default='low')
        Cardinality level: 'low' or 'high'.

    Returns
    -------
    Pipeline
        Sklearn pipeline with imputer and encoder.

    Raises
    ------
    ValueError
        If *encoder_type* is not one of the recognised values.
    """
    imputer = SimpleImputer(strategy="constant", fill_value="missing")

    if encoder_type == "onehot":
        if cardinality == "low":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
    elif encoder_type == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    elif encoder_type == "target":
        if not CATEGORY_ENCODERS_AVAILABLE:
            logger.warning(
                "category_encoders not installed. Falling back to ordinal encoding. "
                "Install with: pip install category_encoders"
            )
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        else:
            encoder = TargetEncoder(handle_unknown="value", handle_missing="value")
    elif encoder_type == "binary":
        if not CATEGORY_ENCODERS_AVAILABLE:
            logger.warning(
                "category_encoders not installed. Falling back to onehot encoding. "
                "Install with: pip install category_encoders"
            )
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = BinaryEncoder(handle_unknown="value", handle_missing="value")
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Choose from {_VALID_ENCODERS!r}"
        )

    return Pipeline(steps=[("imputer", imputer), ("encoding", encoder)])


class TimeoutException(Exception):
    """Raised when a code block exceeds the allotted time limit."""


@contextmanager
def time_limit(seconds: int):
    """Context manager to limit execution time of a code block.

    Parameters
    ----------
    seconds : int
        Maximum time in seconds for the code block to execute.

    Raises
    ------
    TimeoutException
        If the code block exceeds the time limit.
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows: signal.alarm is not available — yield without timeout
        # The fit() method will track time manually as a fallback
        yield


def get_card_split(
    df: pd.DataFrame, cols: pd.Index, n: int = 11
) -> Tuple[pd.Index, pd.Index]:
    """Split categorical columns into two lists based on cardinality.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : array-like
        Categorical columns to evaluate.
    n : int, optional (default=11)
        Columns with more than *n* unique values are considered high cardinality.

    Returns
    -------
    card_low : pandas.Index
        Columns with cardinality <= *n*.
    card_high : pandas.Index
        Columns with cardinality > *n*.
    """
    cols = pd.Index(cols)
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def adjusted_rsquared(r2: float, n: int, p: int) -> float:
    """Calculate adjusted R-squared.

    Parameters
    ----------
    r2 : float
        R-squared value.
    n : int
        Number of observations.
    p : int
        Number of predictors.

    Returns
    -------
    float
        Adjusted R-squared value.
    """
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_init_params(
    cv: Optional[int],
    timeout: Optional[int],
    categorical_encoder: str,
    custom_metric: Optional[Callable],
) -> None:
    """Validate constructor parameters shared by LazyClassifier and LazyRegressor."""
    if cv is not None:
        if not isinstance(cv, int) or cv < 2:
            raise ValueError(f"cv must be an integer >= 2, got {cv!r}")
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {timeout!r}")
    if categorical_encoder not in _VALID_ENCODERS:
        raise ValueError(
            f"categorical_encoder must be one of {_VALID_ENCODERS!r}, "
            f"got {categorical_encoder!r}"
        )
    if custom_metric is not None and not callable(custom_metric):
        raise TypeError(f"custom_metric must be callable, got {type(custom_metric)}")


def _validate_fit_inputs(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
) -> None:
    """Validate shapes and types of data passed to fit()."""
    if hasattr(X_train, "shape") and hasattr(y_train, "shape"):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train has {X_train.shape[0]} samples but y_train has "
                f"{y_train.shape[0]} samples"
            )
    if hasattr(X_test, "shape") and hasattr(y_test, "shape"):
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"X_test has {X_test.shape[0]} samples but y_test has "
                f"{y_test.shape[0]} samples"
            )
    if hasattr(X_train, "shape") and hasattr(X_test, "shape"):
        if len(X_train.shape) > 1 and len(X_test.shape) > 1:
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(
                    f"X_train has {X_train.shape[1]} features but X_test has "
                    f"{X_test.shape[1]} features"
                )
    # Check for empty data
    if hasattr(X_train, "__len__") and len(X_train) == 0:
        raise ValueError("X_train is empty")
    if hasattr(X_test, "__len__") and len(X_test) == 0:
        raise ValueError("X_test is empty")


def _prepare_dataframes(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert ndarrays to DataFrames with consistent column names."""
    if isinstance(X_train, np.ndarray):
        cols = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)
    return X_train, X_test


def _build_preprocessor(
    X_train: pd.DataFrame, categorical_encoder: str
) -> ColumnTransformer:
    """Build a ColumnTransformer for the given data."""
    numeric_features = X_train.select_dtypes(include=[np.number, "bool"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    categorical_low, categorical_high = get_card_split(X_train, categorical_features)

    cat_low = get_categorical_encoder(categorical_encoder, cardinality="low")
    cat_high = get_categorical_encoder(categorical_encoder, cardinality="high")

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical_low", cat_low, categorical_low),
            ("categorical_high", cat_high, categorical_high),
        ]
    )


# ---------------------------------------------------------------------------
# LazyClassifier
# ---------------------------------------------------------------------------


class LazyClassifier:
    """Fit all classification algorithms available in scikit-learn and benchmark them.

    Parameters
    ----------
    verbose : int, optional (default=0)
        Set to a positive number to enable progress bars and per-model metric output.
    ignore_warnings : bool, optional (default=True)
        When True, warnings and errors from individual models are suppressed.
    custom_metric : callable or None, optional (default=None)
        A function ``f(y_true, y_pred)`` used for additional evaluation.
    predictions : bool, optional (default=False)
        When True, ``fit()`` returns a tuple of (scores, predictions_dataframe).
    random_state : int, optional (default=42)
        Random seed passed to models that accept it.
    classifiers : list or ``"all"``, optional (default="all")
        Specific classifier classes to train, or ``"all"`` for every available one.
    cv : int or None, optional (default=None)
        Number of folds for cross-validation.  If None, uses train/test split only.
    timeout : int or float or None, optional (default=None)
        Maximum seconds for each model.  Models exceeding this are skipped.
    categorical_encoder : str, optional (default='onehot')
        Encoder for categorical features: ``'onehot'``, ``'ordinal'``,
        ``'target'``, or ``'binary'``.

    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y = data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=123)
    >>> clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    >>> models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        classifiers: Union[str, List] = "all",
        cv: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        categorical_encoder: str = "onehot",
    ):
        _validate_init_params(cv, timeout, categorical_encoder, custom_metric)
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models: Dict[str, Pipeline] = {}
        self.errors: Dict[str, Exception] = {}
        self.random_state = random_state
        self.classifiers = classifiers
        self.cv = cv
        self.timeout = timeout
        self.categorical_encoder = categorical_encoder
        self.mlflow_enabled = setup_mlflow()

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fit classification algorithms and score on test data.

        Parameters
        ----------
        X_train : array-like
            Training feature matrix.
        X_test : array-like
            Testing feature matrix.
        y_train : array-like
            Training target vector.
        y_test : array-like
            Testing target vector.

        Returns
        -------
        scores : pandas.DataFrame
            Metrics for every model, sorted by Balanced Accuracy.
        predictions : pandas.DataFrame
            Only returned when ``self.predictions`` is True.
        """
        _validate_fit_inputs(X_train, X_test, y_train, y_test)
        X_train, X_test = _prepare_dataframes(X_train, X_test)
        preprocessor = _build_preprocessor(X_train, self.categorical_encoder)

        # Resolve classifier list
        if self.classifiers == "all":
            classifier_list = list(CLASSIFIERS)
        else:
            try:
                classifier_list = [
                    (cls.__name__, cls) for cls in self.classifiers
                ]
            except Exception as exc:
                logger.error("Invalid classifier(s): %s", exc)
                raise ValueError(f"Invalid classifier(s): {exc}") from exc

        # Metric accumulators
        names: List[str] = []
        accuracy_list: List[float] = []
        b_accuracy_list: List[float] = []
        roc_auc_list: List[Optional[float]] = []
        f1_list: List[float] = []
        precision_list: List[float] = []
        recall_list: List[float] = []
        time_list: List[float] = []
        predictions_dict: Dict[str, np.ndarray] = {}
        custom_metric_list: List[Optional[float]] = []

        # CV accumulators
        cv_data: Dict[str, List[Optional[float]]] = {
            key: []
            for key in [
                "Accuracy CV Mean", "Accuracy CV Std",
                "Balanced Accuracy CV Mean", "Balanced Accuracy CV Std",
                "ROC AUC CV Mean", "ROC AUC CV Std",
                "F1 Score CV Mean", "F1 Score CV Std",
                "Precision CV Mean", "Precision CV Std",
                "Recall CV Mean", "Recall CV Std",
            ]
        }

        progress_bar = notebook_tqdm if _use_notebook_tqdm else tqdm
        for name, model in progress_bar(classifier_list, disable=(self.verbose == 0)):
            start = time.time()
            mlflow_active_run = None
            try:
                if self.mlflow_enabled and MLFLOW_AVAILABLE:
                    mlflow_active_run = mlflow.start_run(run_name=f"LazyClassifier-{name}")
                    mlflow.log_param("model_name", name)

                if "random_state" in model().get_params():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", model())]
                    )

                pipe.fit(X_train, y_train)
                fit_time = time.time() - start

                if self.timeout and fit_time > self.timeout:
                    logger.info(
                        "%s exceeded timeout (%.2fs > %ss), skipping...",
                        name, fit_time, self.timeout,
                    )
                    if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                        mlflow.end_run()
                    continue

                self.models[name] = pipe

                # Cross-validation
                if self.cv:
                    try:
                        scoring = {
                            "accuracy": "accuracy",
                            "balanced_accuracy": "balanced_accuracy",
                            "f1_weighted": "f1_weighted",
                            "precision_weighted": "precision_weighted",
                            "recall_weighted": "recall_weighted",
                            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
                        }
                        cv_results = cross_validate(
                            pipe, X_train, y_train,
                            cv=self.cv, scoring=scoring,
                            n_jobs=-1, error_score="raise",
                        )
                        cv_data["Accuracy CV Mean"].append(cv_results["test_accuracy"].mean())
                        cv_data["Accuracy CV Std"].append(cv_results["test_accuracy"].std())
                        cv_data["Balanced Accuracy CV Mean"].append(cv_results["test_balanced_accuracy"].mean())
                        cv_data["Balanced Accuracy CV Std"].append(cv_results["test_balanced_accuracy"].std())
                        cv_data["F1 Score CV Mean"].append(cv_results["test_f1_weighted"].mean())
                        cv_data["F1 Score CV Std"].append(cv_results["test_f1_weighted"].std())
                        cv_data["Precision CV Mean"].append(cv_results["test_precision_weighted"].mean())
                        cv_data["Precision CV Std"].append(cv_results["test_precision_weighted"].std())
                        cv_data["Recall CV Mean"].append(cv_results["test_recall_weighted"].mean())
                        cv_data["Recall CV Std"].append(cv_results["test_recall_weighted"].std())
                        try:
                            cv_data["ROC AUC CV Mean"].append(cv_results["test_roc_auc_ovr_weighted"].mean())
                            cv_data["ROC AUC CV Std"].append(cv_results["test_roc_auc_ovr_weighted"].std())
                        except Exception:
                            cv_data["ROC AUC CV Mean"].append(None)
                            cv_data["ROC AUC CV Std"].append(None)
                    except Exception as cv_exc:
                        if not self.ignore_warnings:
                            logger.warning("Cross-validation failed for %s: %s", name, cv_exc)
                        for key in cv_data:
                            cv_data[key].append(None)

                # Test-set metrics
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted")

                try:
                    if hasattr(pipe, "predict_proba"):
                        y_pred_proba = pipe.predict_proba(X_test)
                        if y_pred_proba.shape[1] == 2:
                            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            roc_auc = roc_auc_score(
                                y_test, y_pred_proba, multi_class="ovr", average="weighted"
                            )
                    elif hasattr(pipe, "decision_function"):
                        roc_auc = roc_auc_score(y_test, pipe.decision_function(X_test))
                    else:
                        roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as roc_exc:
                    roc_auc = None
                    if not self.ignore_warnings:
                        logger.warning("ROC AUC couldn't be calculated for %s: %s", name, roc_exc)

                # MLflow logging
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("balanced_accuracy", b_accuracy)
                    mlflow.log_metric("f1_score", f1)
                    if roc_auc is not None:
                        mlflow.log_metric("roc_auc", roc_auc)
                    mlflow.log_metric("training_time", time.time() - start)
                    try:
                        signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
                        mlflow.sklearn.log_model(
                            pipe, f"{name}_model",
                            signature=signature,
                            registered_model_name=f"lazy_classifier_{name}",
                        )
                    except Exception as mlflow_exc:
                        if not self.ignore_warnings:
                            logger.warning("Failed to log model %s to MLflow: %s", name, mlflow_exc)

                names.append(name)
                accuracy_list.append(accuracy)
                b_accuracy_list.append(b_accuracy)
                roc_auc_list.append(roc_auc)
                f1_list.append(f1)
                precision_list.append(precision)
                recall_list.append(recall)
                time_list.append(time.time() - start)

                if self.custom_metric is not None:
                    try:
                        custom_val = self.custom_metric(y_test, y_pred)
                        custom_metric_list.append(custom_val)
                        if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                            mlflow.log_metric(self.custom_metric.__name__, custom_val)
                    except Exception as custom_exc:
                        custom_metric_list.append(None)
                        if not self.ignore_warnings:
                            logger.warning(
                                "Custom metric %s failed for %s: %s",
                                self.custom_metric.__name__, name, custom_exc,
                            )

                if self.verbose > 0:
                    logger.info(
                        "Model=%s Accuracy=%.4f BalAcc=%.4f ROC_AUC=%s F1=%.4f Time=%.2fs",
                        name, accuracy, b_accuracy, roc_auc, f1, time.time() - start,
                    )

                if self.predictions:
                    predictions_dict[name] = y_pred

                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()

            except Exception as exc:
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()
                self.errors[name] = exc
                if not self.ignore_warnings:
                    logger.warning("%s model failed to execute: %s", name, exc)

        # Build results DataFrame
        scores_dict: Dict[str, Any] = {
            "Model": names,
            "Accuracy": accuracy_list,
            "Balanced Accuracy": b_accuracy_list,
            "ROC AUC": roc_auc_list,
            "F1 Score": f1_list,
            "Precision": precision_list,
            "Recall": recall_list,
        }

        if self.custom_metric is not None:
            scores_dict[self.custom_metric.__name__] = custom_metric_list

        if self.cv:
            scores_dict.update(cv_data)

        scores_dict["Time Taken"] = time_list

        scores = pd.DataFrame(scores_dict)
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index("Model")

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions_dict)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Pipeline]:
        """Return all trained model pipelines.

        If ``fit()`` has not been called yet, it will be invoked automatically.

        Parameters
        ----------
        X_train, X_test, y_train, y_test : array-like
            Training and testing data.

        Returns
        -------
        dict
            Mapping of model name to fitted Pipeline.
        """
        if len(self.models) == 0:
            self.fit(X_train, X_test, y_train, y_test)
        return self.models

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        model_name: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Make predictions using fitted models.

        Parameters
        ----------
        X_test : array-like
            Test feature matrix.
        model_name : str or None, optional (default=None)
            Specific model to use.  If None, returns predictions from all models.

        Returns
        -------
        dict or numpy.ndarray
            Dictionary of predictions keyed by model name, or a single array.

        Raises
        ------
        ValueError
            If no models have been fitted or the model name is unknown.
        """
        if len(self.models) == 0:
            raise ValueError("No models have been fitted yet. Please call fit() first.")

        if isinstance(X_test, np.ndarray):
            cols = [f"feature_{i}" for i in range(X_test.shape[1])]
            X_test = pd.DataFrame(X_test, columns=cols)

        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available models: {list(self.models.keys())}"
                )
            return self.models[model_name].predict(X_test)

        return {name: model.predict(X_test) for name, model in self.models.items()}


# ---------------------------------------------------------------------------
# LazyRegressor
# ---------------------------------------------------------------------------


class LazyRegressor:
    """Fit all regression algorithms available in scikit-learn and benchmark them.

    Parameters
    ----------
    verbose : int, optional (default=0)
        Set to a positive number to enable progress bars and per-model metric output.
    ignore_warnings : bool, optional (default=True)
        When True, warnings and errors from individual models are suppressed.
    custom_metric : callable or None, optional (default=None)
        A function ``f(y_true, y_pred)`` used for additional evaluation.
    predictions : bool, optional (default=False)
        When True, ``fit()`` returns a tuple of (scores, predictions_dataframe).
    random_state : int, optional (default=42)
        Random seed passed to models that accept it.
    regressors : list or ``"all"``, optional (default="all")
        Specific regressor classes to train, or ``"all"`` for every available one.
    cv : int or None, optional (default=None)
        Number of folds for cross-validation.  If None, uses train/test split only.
    timeout : int or float or None, optional (default=None)
        Maximum seconds for each model.  Models exceeding this are skipped.
    categorical_encoder : str, optional (default='onehot')
        Encoder for categorical features: ``'onehot'``, ``'ordinal'``,
        ``'target'``, or ``'binary'``.

    Examples
    --------
    >>> from lazypredict.Supervised import LazyRegressor
    >>> from sklearn import datasets
    >>> from sklearn.utils import shuffle
    >>> import numpy as np
    >>> diabetes = datasets.load_diabetes()
    >>> X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
    >>> X = X.astype(np.float32)
    >>> offset = int(X.shape[0] * 0.9)
    >>> X_train, y_train = X[:offset], y[:offset]
    >>> X_test, y_test = X[offset:], y[offset:]
    >>> reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    >>> models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        regressors: Union[str, List] = "all",
        cv: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        categorical_encoder: str = "onehot",
    ):
        _validate_init_params(cv, timeout, categorical_encoder, custom_metric)
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models: Dict[str, Pipeline] = {}
        self.errors: Dict[str, Exception] = {}
        self.random_state = random_state
        self.regressors = regressors
        self.cv = cv
        self.timeout = timeout
        self.categorical_encoder = categorical_encoder
        self.mlflow_enabled = setup_mlflow()

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fit regression algorithms and score on test data.

        Parameters
        ----------
        X_train : array-like
            Training feature matrix.
        X_test : array-like
            Testing feature matrix.
        y_train : array-like
            Training target vector.
        y_test : array-like
            Testing target vector.

        Returns
        -------
        scores : pandas.DataFrame
            Metrics for every model, sorted by Adjusted R-Squared.
        predictions : pandas.DataFrame
            Only returned when ``self.predictions`` is True.
        """
        _validate_fit_inputs(X_train, X_test, y_train, y_test)
        X_train, X_test = _prepare_dataframes(X_train, X_test)
        preprocessor = _build_preprocessor(X_train, self.categorical_encoder)

        # Resolve regressor list
        if self.regressors == "all":
            regressor_list = list(REGRESSORS)
        else:
            try:
                regressor_list = [
                    (cls.__name__, cls) for cls in self.regressors
                ]
            except Exception as exc:
                logger.error("Invalid regressor(s): %s", exc)
                raise ValueError(f"Invalid regressor(s): {exc}") from exc

        # Metric accumulators
        names: List[str] = []
        r2_list: List[float] = []
        adjr2_list: List[float] = []
        rmse_list: List[float] = []
        time_list: List[float] = []
        predictions_dict: Dict[str, np.ndarray] = {}
        custom_metric_list: List[Optional[float]] = []

        # CV accumulators
        cv_data: Dict[str, List[Optional[float]]] = {
            key: []
            for key in [
                "R-Squared CV Mean", "R-Squared CV Std",
                "Adjusted R-Squared CV Mean", "Adjusted R-Squared CV Std",
                "RMSE CV Mean", "RMSE CV Std",
            ]
        }

        progress_bar = notebook_tqdm if _use_notebook_tqdm else tqdm
        for name, model in progress_bar(regressor_list, disable=(self.verbose == 0)):
            start = time.time()
            mlflow_active_run = None
            try:
                if self.mlflow_enabled and MLFLOW_AVAILABLE:
                    mlflow_active_run = mlflow.start_run(run_name=f"LazyRegressor-{name}")
                    mlflow.log_param("model_name", name)

                if "random_state" in model().get_params():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("regressor", model())]
                    )

                pipe.fit(X_train, y_train)
                fit_time = time.time() - start

                if self.timeout and fit_time > self.timeout:
                    logger.info(
                        "%s exceeded timeout (%.2fs > %ss), skipping...",
                        name, fit_time, self.timeout,
                    )
                    if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                        mlflow.end_run()
                    continue

                self.models[name] = pipe

                # Cross-validation
                if self.cv:
                    try:
                        scoring = {
                            "r2": "r2",
                            "neg_mean_squared_error": "neg_mean_squared_error",
                        }
                        cv_results = cross_validate(
                            pipe, X_train, y_train,
                            cv=self.cv, scoring=scoring,
                            n_jobs=-1, error_score="raise",
                        )
                        cv_data["R-Squared CV Mean"].append(cv_results["test_r2"].mean())
                        cv_data["R-Squared CV Std"].append(cv_results["test_r2"].std())

                        rmse_cv = np.sqrt(-cv_results["test_neg_mean_squared_error"])
                        cv_data["RMSE CV Mean"].append(rmse_cv.mean())
                        cv_data["RMSE CV Std"].append(rmse_cv.std())

                        adj_r2_cv = [
                            adjusted_rsquared(r2, X_train.shape[0], X_train.shape[1])
                            for r2 in cv_results["test_r2"]
                        ]
                        cv_data["Adjusted R-Squared CV Mean"].append(np.mean(adj_r2_cv))
                        cv_data["Adjusted R-Squared CV Std"].append(np.std(adj_r2_cv))
                    except Exception as cv_exc:
                        if not self.ignore_warnings:
                            logger.warning("Cross-validation failed for %s: %s", name, cv_exc)
                        for key in cv_data:
                            cv_data[key].append(None)

                # Test-set metrics
                y_pred = pipe.predict(X_test)
                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # MLflow logging
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.log_metric("r_squared", r_squared)
                    mlflow.log_metric("adjusted_r_squared", adj_rsquared)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("training_time", time.time() - start)
                    try:
                        signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
                        mlflow.sklearn.log_model(
                            pipe, f"{name}_model",
                            signature=signature,
                            registered_model_name=f"lazy_regressor_{name}",
                        )
                    except Exception as mlflow_exc:
                        if not self.ignore_warnings:
                            logger.warning("Failed to log model %s to MLflow: %s", name, mlflow_exc)

                names.append(name)
                r2_list.append(r_squared)
                adjr2_list.append(adj_rsquared)
                rmse_list.append(rmse)
                time_list.append(time.time() - start)

                if self.custom_metric:
                    try:
                        custom_val = self.custom_metric(y_test, y_pred)
                        custom_metric_list.append(custom_val)
                        if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                            mlflow.log_metric(self.custom_metric.__name__, custom_val)
                    except Exception as custom_exc:
                        custom_metric_list.append(None)
                        if not self.ignore_warnings:
                            logger.warning(
                                "Custom metric %s failed for %s: %s",
                                self.custom_metric.__name__, name, custom_exc,
                            )

                if self.verbose > 0:
                    logger.info(
                        "Model=%s R2=%.4f AdjR2=%.4f RMSE=%.4f Time=%.2fs",
                        name, r_squared, adj_rsquared, rmse, time.time() - start,
                    )

                if self.predictions:
                    predictions_dict[name] = y_pred

                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()

            except Exception as exc:
                if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_active_run:
                    mlflow.end_run()
                self.errors[name] = exc
                if not self.ignore_warnings:
                    logger.warning("%s model failed to execute: %s", name, exc)

        # Build results DataFrame
        scores_dict: Dict[str, Any] = {
            "Model": names,
            "Adjusted R-Squared": adjr2_list,
            "R-Squared": r2_list,
            "RMSE": rmse_list,
        }

        if self.cv:
            scores_dict.update(cv_data)

        if self.custom_metric:
            scores_dict[self.custom_metric.__name__] = custom_metric_list

        scores_dict["Time Taken"] = time_list

        scores = pd.DataFrame(scores_dict)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index("Model")

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions_dict)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Pipeline]:
        """Return all trained model pipelines.

        If ``fit()`` has not been called yet, it will be invoked automatically.

        Parameters
        ----------
        X_train, X_test, y_train, y_test : array-like
            Training and testing data.

        Returns
        -------
        dict
            Mapping of model name to fitted Pipeline.
        """
        if len(self.models) == 0:
            self.fit(X_train, X_test, y_train, y_test)
        return self.models

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        model_name: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Make predictions using fitted models.

        Parameters
        ----------
        X_test : array-like
            Test feature matrix.
        model_name : str or None, optional (default=None)
            Specific model to use.  If None, returns predictions from all models.

        Returns
        -------
        dict or numpy.ndarray
            Dictionary of predictions keyed by model name, or a single array.

        Raises
        ------
        ValueError
            If no models have been fitted or the model name is unknown.
        """
        if len(self.models) == 0:
            raise ValueError("No models have been fitted yet. Please call fit() first.")

        if isinstance(X_test, np.ndarray):
            cols = [f"feature_{i}" for i in range(X_test.shape[1])]
            X_test = pd.DataFrame(X_test, columns=cols)

        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available models: {list(self.models.keys())}"
                )
            return self.models[model_name].predict(X_test)

        return {name: model.predict(X_test) for name, model in self.models.items()}


# Backward-compatible aliases
Regression = LazyRegressor
Classification = LazyClassifier
