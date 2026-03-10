"""
Supervised Models — LazyClassifier and LazyRegressor for rapid model benchmarking.

Provides LazyClassifier and LazyRegressor classes that train multiple
scikit-learn models with minimal code to quickly identify which algorithms
perform best on a given dataset.
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import logging
import signal
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
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
from sklearn.utils import all_estimators

# Re-export from submodules for backward compatibility
from lazypredict._base import (
    LazyEstimator,
    _validate_fit_inputs,
    _validate_init_params,
)
from lazypredict.config import (
    REMOVED_CLASSIFIERS,
    REMOVED_REGRESSORS,
    VALID_ENCODERS,
)
from lazypredict.exceptions import ModelFitError, TimeoutException
from lazypredict.integrations.mlflow import (
    MLFLOW_AVAILABLE,
    is_mlflow_tracking_enabled,
    setup_mlflow,
)
from lazypredict.metrics import adjusted_rsquared
from lazypredict.preprocessing import (
    CATEGORY_ENCODERS_AVAILABLE,
    build_preprocessor,
    categorical_transformer_high,
    categorical_transformer_low,
    get_card_split,
    get_categorical_encoder,
    numeric_transformer,
    prepare_dataframes,
)

# Module-level logger — users can configure via logging.getLogger("lazypredict")
logger = logging.getLogger("lazypredict")

# Keep private aliases used in the module
_VALID_ENCODERS = VALID_ENCODERS
_REMOVED_CLASSIFIERS = REMOVED_CLASSIFIERS
_REMOVED_REGRESSORS = REMOVED_REGRESSORS

# Private aliases for prepare/build used within fit()
_prepare_dataframes = prepare_dataframes
_build_preprocessor = build_preprocessor

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

# Optional catboost
try:
    import catboost
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

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

# Optional MLflow
try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

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

if _CATBOOST_AVAILABLE:
    REGRESSORS.append(("CatBoostRegressor", catboost.CatBoostRegressor))
    CLASSIFIERS.append(("CatBoostClassifier", catboost.CatBoostClassifier))

if PERPETUAL_AVAILABLE:
    REGRESSORS.append(("PerpetualBooster", PerpetualBooster))
    CLASSIFIERS.append(("PerpetualBooster", PerpetualBooster))

# Backward-compatible aliases for removed_ lists
removed_classifiers = list(_REMOVED_CLASSIFIERS)
removed_regressors = list(_REMOVED_REGRESSORS)


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
        yield


# ---------------------------------------------------------------------------
# LazyClassifier
# ---------------------------------------------------------------------------


class LazyClassifier(LazyEstimator):
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
    n_jobs : int, optional (default=-1)
        Number of parallel jobs for cross-validation. -1 uses all processors.
    max_models : int or None, optional (default=None)
        Maximum number of models to train. None means train all.
    progress_callback : callable or None, optional (default=None)
        Callback ``f(model_name, current, total, metrics)`` called after each model.
    use_gpu : bool, optional (default=False)
        When True, enables GPU acceleration for models that support it
        (e.g., XGBoost, LightGBM, CatBoost). When cuML (RAPIDS) is installed,
        GPU-accelerated scikit-learn equivalents are also added automatically.
        Falls back to CPU if CUDA is unavailable.

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
        n_jobs: int = -1,
        max_models: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        use_gpu: bool = False,
    ):
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            cv=cv,
            timeout=timeout,
            categorical_encoder=categorical_encoder,
            n_jobs=n_jobs,
            max_models=max_models,
            progress_callback=progress_callback,
            use_gpu=use_gpu,
        )
        self.classifiers = classifiers

    def _estimator_step_name(self) -> str:
        return "classifier"

    def _get_estimator_list(self) -> List[Tuple[str, Any]]:
        if self.classifiers == "all":
            return list(CLASSIFIERS)
        try:
            return [(cls.__name__, cls) for cls in self.classifiers]
        except Exception as exc:
            logger.error("Invalid classifier(s): %s", exc)
            raise ValueError(f"Invalid classifier(s): {exc}") from exc

    def _cv_scoring(self) -> Optional[Dict[str, str]]:
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        }

    def _cv_column_names(self) -> List[str]:
        return [
            "Accuracy CV Mean", "Accuracy CV Std",
            "Balanced Accuracy CV Mean", "Balanced Accuracy CV Std",
            "ROC AUC CV Mean", "ROC AUC CV Std",
            "F1 Score CV Mean", "F1 Score CV Std",
            "Precision CV Mean", "Precision CV Std",
            "Recall CV Mean", "Recall CV Std",
        ]

    def _process_cv_results(
        self, cv_results: Dict[str, Any], X_train: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {
            "Accuracy CV Mean": cv_results["test_accuracy"].mean(),
            "Accuracy CV Std": cv_results["test_accuracy"].std(),
            "Balanced Accuracy CV Mean": cv_results["test_balanced_accuracy"].mean(),
            "Balanced Accuracy CV Std": cv_results["test_balanced_accuracy"].std(),
            "F1 Score CV Mean": cv_results["test_f1_weighted"].mean(),
            "F1 Score CV Std": cv_results["test_f1_weighted"].std(),
            "Precision CV Mean": cv_results["test_precision_weighted"].mean(),
            "Precision CV Std": cv_results["test_precision_weighted"].std(),
            "Recall CV Mean": cv_results["test_recall_weighted"].mean(),
            "Recall CV Std": cv_results["test_recall_weighted"].std(),
        }
        try:
            result["ROC AUC CV Mean"] = cv_results["test_roc_auc_ovr_weighted"].mean()
            result["ROC AUC CV Std"] = cv_results["test_roc_auc_ovr_weighted"].std()
        except Exception:
            result["ROC AUC CV Mean"] = None
            result["ROC AUC CV Std"] = None
        return result

    def _compute_metrics(
        self, pipe: Pipeline, X_test: pd.DataFrame, y_test: Any, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        y_pred = pipe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        b_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall_val = recall_score(y_test, y_pred, average="weighted")

        roc_auc = None
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
            if not self.ignore_warnings:
                logger.warning("ROC AUC couldn't be calculated: %s", roc_exc)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": b_accuracy,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall_val,
        }

    def _build_scores_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            row: Dict[str, Any] = {
                "Model": r["name"],
                "Accuracy": r["accuracy"],
                "Balanced Accuracy": r["balanced_accuracy"],
                "ROC AUC": r["roc_auc"],
                "F1 Score": r["f1"],
                "Precision": r["precision"],
                "Recall": r["recall"],
            }
            if self.custom_metric is not None:
                row[self.custom_metric.__name__] = r.get("custom_metric")
            # CV columns
            for col in self._cv_column_names():
                if col in r:
                    row[col] = r[col]
            row["Time Taken"] = r["time"]
            rows.append(row)

        scores = pd.DataFrame(rows)
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index("Model")
        return scores

    def _log_verbose(self, name: str, metrics: Dict[str, Any]) -> None:
        logger.info(
            "Model=%s Accuracy=%.4f BalAcc=%.4f ROC_AUC=%s F1=%.4f Time=%.2fs",
            name,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["roc_auc"],
            metrics["f1"],
            metrics["time"],
        )


# ---------------------------------------------------------------------------
# LazyRegressor
# ---------------------------------------------------------------------------


class LazyRegressor(LazyEstimator):
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
    n_jobs : int, optional (default=-1)
        Number of parallel jobs for cross-validation. -1 uses all processors.
    max_models : int or None, optional (default=None)
        Maximum number of models to train. None means train all.
    progress_callback : callable or None, optional (default=None)
        Callback ``f(model_name, current, total, metrics)`` called after each model.
    use_gpu : bool, optional (default=False)
        When True, enables GPU acceleration for models that support it
        (e.g., XGBoost, LightGBM). Falls back to CPU if CUDA is unavailable.

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
        n_jobs: int = -1,
        max_models: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        use_gpu: bool = False,
    ):
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            cv=cv,
            timeout=timeout,
            categorical_encoder=categorical_encoder,
            n_jobs=n_jobs,
            max_models=max_models,
            progress_callback=progress_callback,
            use_gpu=use_gpu,
        )
        self.regressors = regressors

    def _estimator_step_name(self) -> str:
        return "regressor"

    def _get_estimator_list(self) -> List[Tuple[str, Any]]:
        if self.regressors == "all":
            return list(REGRESSORS)
        try:
            return [(cls.__name__, cls) for cls in self.regressors]
        except Exception as exc:
            logger.error("Invalid regressor(s): %s", exc)
            raise ValueError(f"Invalid regressor(s): {exc}") from exc

    def _cv_scoring(self) -> Optional[Dict[str, str]]:
        return {
            "r2": "r2",
            "neg_mean_squared_error": "neg_mean_squared_error",
        }

    def _cv_column_names(self) -> List[str]:
        return [
            "R-Squared CV Mean", "R-Squared CV Std",
            "Adjusted R-Squared CV Mean", "Adjusted R-Squared CV Std",
            "RMSE CV Mean", "RMSE CV Std",
        ]

    def _process_cv_results(
        self, cv_results: Dict[str, Any], X_train: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        rmse_cv = np.sqrt(-cv_results["test_neg_mean_squared_error"])
        adj_r2_cv = [
            adjusted_rsquared(r2, X_train.shape[0], X_train.shape[1])
            for r2 in cv_results["test_r2"]
        ]
        return {
            "R-Squared CV Mean": cv_results["test_r2"].mean(),
            "R-Squared CV Std": cv_results["test_r2"].std(),
            "Adjusted R-Squared CV Mean": np.mean(adj_r2_cv),
            "Adjusted R-Squared CV Std": np.std(adj_r2_cv),
            "RMSE CV Mean": rmse_cv.mean(),
            "RMSE CV Std": rmse_cv.std(),
        }

    def _compute_metrics(
        self, pipe: Pipeline, X_test: pd.DataFrame, y_test: Any, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        y_pred = pipe.predict(X_test)
        r_squared = r2_score(y_test, y_pred)
        adj_rsquared = adjusted_rsquared(
            r_squared, X_test.shape[0], X_test.shape[1]
        )
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {
            "r_squared": r_squared,
            "adjusted_r_squared": adj_rsquared,
            "rmse": rmse,
        }

    def _build_scores_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            row: Dict[str, Any] = {
                "Model": r["name"],
                "Adjusted R-Squared": r["adjusted_r_squared"],
                "R-Squared": r["r_squared"],
                "RMSE": r["rmse"],
            }
            # CV columns
            for col in self._cv_column_names():
                if col in r:
                    row[col] = r[col]
            if self.custom_metric is not None:
                row[self.custom_metric.__name__] = r.get("custom_metric")
            row["Time Taken"] = r["time"]
            rows.append(row)

        scores = pd.DataFrame(rows)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index("Model")
        return scores

    def _log_verbose(self, name: str, metrics: Dict[str, Any]) -> None:
        logger.info(
            "Model=%s R2=%.4f AdjR2=%.4f RMSE=%.4f Time=%.2fs",
            name,
            metrics["r_squared"],
            metrics["adjusted_r_squared"],
            metrics["rmse"],
            metrics["time"],
        )


# Backward-compatible aliases
Regression = LazyRegressor
Classification = LazyClassifier
