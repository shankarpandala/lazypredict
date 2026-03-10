"""Base class for LazyClassifier and LazyRegressor — shared logic lives here."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from lazypredict.config import (
    VALID_ENCODERS,
    get_cuml_models,
    get_gpu_model_params,
    is_gpu_available,
)
from lazypredict.exceptions import ModelFitError  # noqa: F401
from lazypredict.integrations.mlflow import MLFLOW_AVAILABLE, setup_mlflow
from lazypredict.preprocessing import build_preprocessor, prepare_dataframes

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

# Optional MLflow
try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None  # type: ignore[assignment]


def _validate_init_params(
    cv: Optional[int],
    timeout: Optional[Union[int, float]],
    categorical_encoder: str,
    custom_metric: Optional[Callable],
    n_jobs: Optional[int],
) -> None:
    """Validate constructor parameters shared by LazyClassifier and LazyRegressor."""
    if cv is not None:
        if not isinstance(cv, int) or cv < 2:
            raise ValueError(f"cv must be an integer >= 2, got {cv!r}")
    if timeout is not None:
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {timeout!r}")
    if categorical_encoder not in VALID_ENCODERS:
        raise ValueError(
            f"categorical_encoder must be one of {VALID_ENCODERS!r}, "
            f"got {categorical_encoder!r}"
        )
    if custom_metric is not None and not callable(custom_metric):
        raise TypeError(f"custom_metric must be callable, got {type(custom_metric)}")
    if n_jobs is not None and not isinstance(n_jobs, int):
        raise ValueError(f"n_jobs must be an integer or None, got {n_jobs!r}")


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
    if hasattr(X_train, "__len__") and len(X_train) == 0:
        raise ValueError("X_train is empty")
    if hasattr(X_test, "__len__") and len(X_test) == 0:
        raise ValueError("X_test is empty")


class LazyEstimator:
    """Abstract base class with shared logic for LazyClassifier and LazyRegressor.

    Subclasses must implement ``_get_estimator_list``, ``_compute_metrics``,
    ``_build_scores_dataframe``, and ``_estimator_step_name``.
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        cv: Optional[int] = None,
        timeout: Optional[Union[int, float]] = None,
        categorical_encoder: str = "onehot",
        n_jobs: int = -1,
        max_models: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        use_gpu: bool = False,
    ):
        _validate_init_params(cv, timeout, categorical_encoder, custom_metric, n_jobs)
        if max_models is not None and (not isinstance(max_models, int) or max_models < 1):
            raise ValueError(f"max_models must be a positive integer, got {max_models!r}")
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models: Dict[str, Pipeline] = {}
        self.errors: Dict[str, Exception] = {}
        self.random_state = random_state
        self.cv = cv
        self.timeout = timeout
        self.categorical_encoder = categorical_encoder
        self.n_jobs = n_jobs
        self.max_models = max_models
        self.progress_callback = progress_callback
        self.use_gpu = use_gpu
        self.mlflow_enabled = setup_mlflow()

        if self.use_gpu:
            if is_gpu_available():
                logger.info("GPU acceleration enabled. CUDA is available.")
            else:
                logger.warning(
                    "GPU requested but CUDA is not available. "
                    "Models that require GPU will fall back to CPU."
                )

    # -- Abstract interface (to be overridden by subclasses) ------------------

    def _get_estimator_list(self) -> List[Tuple[str, Any]]:
        raise NotImplementedError

    def _estimator_step_name(self) -> str:
        raise NotImplementedError

    def _compute_metrics(
        self, pipe: Pipeline, X_test: pd.DataFrame, y_test: Any, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_scores_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        raise NotImplementedError

    def _cv_scoring(self) -> Optional[Dict[str, str]]:
        raise NotImplementedError

    def _process_cv_results(
        self, cv_results: Dict[str, Any], X_train: pd.DataFrame
    ) -> Dict[str, Optional[float]]:
        raise NotImplementedError

    # -- Shared fit logic -----------------------------------------------------

    def fit(  # noqa: C901
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Fit estimators and score on test data.

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
            Metrics for every model.
        predictions : pandas.DataFrame
            Only returned when ``self.predictions`` is True.
        """
        # Dataset size warnings
        if hasattr(X_train, "shape"):
            n_samples, n_features = X_train.shape
            if n_samples > 100_000:
                logger.warning(
                    "Large dataset detected (%d samples). Training all models may "
                    "take a long time. Consider using a subset or setting max_models/timeout.",
                    n_samples,
                )
            if n_features > 500:
                logger.warning(
                    "High-dimensional dataset (%d features). Some models may be slow "
                    "or fail. Consider dimensionality reduction.",
                    n_features,
                )

        _validate_fit_inputs(X_train, X_test, y_train, y_test)
        X_train, X_test = prepare_dataframes(X_train, X_test)
        preprocessor = build_preprocessor(X_train, self.categorical_encoder)

        estimator_list = self._get_estimator_list()
        step_name = self._estimator_step_name()

        # When GPU requested, append cuML GPU-accelerated models if available
        if self.use_gpu:
            cuml_models = get_cuml_models()
            if cuml_models:
                existing_names = {n for n, _ in estimator_list}
                for cuml_name, cuml_cls in cuml_models.items():
                    if cuml_name not in existing_names:
                        # Only add if compatible with current task type
                        try:
                            from sklearn.base import ClassifierMixin, RegressorMixin
                            if step_name == "classifier" and issubclass(cuml_cls, ClassifierMixin):
                                estimator_list.append((cuml_name, cuml_cls))
                            elif step_name == "regressor" and issubclass(cuml_cls, RegressorMixin):
                                estimator_list.append((cuml_name, cuml_cls))
                            elif step_name not in ("classifier", "regressor"):
                                estimator_list.append((cuml_name, cuml_cls))
                        except TypeError:
                            # cuML classes may not work with issubclass
                            estimator_list.append((cuml_name, cuml_cls))
                logger.info(
                    "cuML (RAPIDS) is available — added %d GPU-accelerated models.",
                    len([n for n in cuml_models if n not in existing_names]),
                )

        # Apply max_models limit
        if self.max_models is not None:
            estimator_list = estimator_list[: self.max_models]

        results: List[Dict[str, Any]] = []
        predictions_dict: Dict[str, np.ndarray] = {}

        progress_bar = notebook_tqdm if _use_notebook_tqdm else tqdm
        total = len(estimator_list)
        for idx, (name, model) in enumerate(
            progress_bar(estimator_list, disable=(self.verbose == 0))
        ):
            start = time.time()
            mlflow_active_run = None
            try:
                if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None:
                    mlflow_active_run = _mlflow.start_run(
                        run_name=f"{self.__class__.__name__}-{name}"
                    )
                    _mlflow.log_param("model_name", name)

                model_kwargs = get_gpu_model_params(model, self.use_gpu)
                if "random_state" in model().get_params():
                    model_kwargs["random_state"] = self.random_state
                # CatBoost: suppress verbose output by default
                module = getattr(model, "__module__", "") or ""
                if "catboost" in module:
                    model_kwargs.setdefault("verbose", 0)
                pipe = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        (step_name, model(**model_kwargs)),
                    ]
                )

                pipe.fit(X_train, y_train)
                fit_time = time.time() - start

                if self.timeout and fit_time > self.timeout:
                    logger.info(
                        "%s exceeded timeout (%.2fs > %ss), skipping...",
                        name, fit_time, self.timeout,
                    )
                    if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None and mlflow_active_run:
                        _mlflow.end_run()
                    continue

                self.models[name] = pipe

                # Cross-validation
                cv_metrics: Dict[str, Optional[float]] = {}
                if self.cv:
                    cv_metrics = self._run_cv(pipe, X_train, y_train, name)

                # Test-set metrics
                metrics = self._compute_metrics(pipe, X_test, y_test, X_train)
                metrics["name"] = name
                metrics["time"] = time.time() - start
                metrics.update(cv_metrics)

                # MLflow logging
                if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None and mlflow_active_run:
                    self._log_mlflow_metrics(metrics, pipe, X_train, name)

                # Custom metric
                if self.custom_metric is not None:
                    y_pred = pipe.predict(X_test)
                    try:
                        custom_val = self.custom_metric(y_test, y_pred)
                        metrics["custom_metric"] = custom_val
                        if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None and mlflow_active_run:
                            _mlflow.log_metric(self.custom_metric.__name__, custom_val)
                    except Exception as custom_exc:
                        metrics["custom_metric"] = None
                        if not self.ignore_warnings:
                            logger.warning(
                                "Custom metric %s failed for %s: %s",
                                self.custom_metric.__name__, name, custom_exc,
                            )

                results.append(metrics)

                if self.verbose > 0:
                    self._log_verbose(name, metrics)

                if self.predictions:
                    predictions_dict[name] = pipe.predict(X_test)

                if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None and mlflow_active_run:
                    _mlflow.end_run()

                # Progress callback
                if self.progress_callback is not None:
                    self.progress_callback(name, idx + 1, total, metrics)

            except Exception as exc:
                if self.mlflow_enabled and MLFLOW_AVAILABLE and _mlflow is not None and mlflow_active_run:
                    _mlflow.end_run()
                self.errors[name] = exc
                if not self.ignore_warnings:
                    logger.warning("%s model failed to execute: %s", name, exc)

                if self.progress_callback is not None:
                    self.progress_callback(name, idx + 1, total, None)

        scores = self._build_scores_dataframe(results)

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions_dict)
            return scores, predictions_df
        return scores, pd.DataFrame()

    def _run_cv(
        self, pipe: Pipeline, X_train: pd.DataFrame, y_train: Any, name: str
    ) -> Dict[str, Optional[float]]:
        """Run cross-validation and return metric dict."""
        from sklearn.model_selection import cross_validate

        scoring = self._cv_scoring()
        if scoring is None:
            return {}
        try:
            cv_results = cross_validate(
                pipe, X_train, y_train,
                cv=self.cv, scoring=scoring,
                n_jobs=self.n_jobs, error_score="raise",
            )
            return self._process_cv_results(cv_results, X_train)
        except Exception as cv_exc:
            if not self.ignore_warnings:
                logger.warning("Cross-validation failed for %s: %s", name, cv_exc)
            return {key: None for key in self._cv_column_names()}

    def _cv_column_names(self) -> List[str]:
        """Return the CV column names for null-filling on failure."""
        raise NotImplementedError

    def _log_verbose(self, name: str, metrics: Dict[str, Any]) -> None:
        """Log per-model metrics when verbose > 0."""
        raise NotImplementedError

    def _log_mlflow_metrics(
        self, metrics: Dict[str, Any], pipe: Pipeline, X_train: pd.DataFrame, name: str
    ) -> None:
        """Log metrics and model to MLflow."""
        if _mlflow is None:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and val is not None:
                _mlflow.log_metric(key, val)
        _mlflow.log_metric("training_time", metrics.get("time", 0))
        try:
            signature = _mlflow.models.infer_signature(X_train, pipe.predict(X_train))
            _mlflow.sklearn.log_model(
                pipe, f"{name}_model",
                signature=signature,
                registered_model_name=f"lazy_{self._estimator_step_name()}_{name}",
            )
        except Exception as mlflow_exc:
            if not self.ignore_warnings:
                logger.warning("Failed to log model %s to MLflow: %s", name, mlflow_exc)

    def provide_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Pipeline]:
        """Return all trained model pipelines.

        If ``fit()`` has not been called yet, it will be invoked automatically.
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
            Specific model to use. If None, returns predictions from all models.

        Returns
        -------
        dict or numpy.ndarray
            Dictionary of predictions keyed by model name, or a single array.
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

    def save_models(self, path: str) -> None:
        """Save all fitted models to disk using joblib.

        Parameters
        ----------
        path : str
            Directory path to save models.
        """
        import os
        import joblib

        if len(self.models) == 0:
            raise ValueError("No models have been fitted yet. Please call fit() first.")

        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            filepath = os.path.join(path, f"{name}.joblib")
            joblib.dump(model, filepath)
            logger.info("Saved %s to %s", name, filepath)

    def load_models(self, path: str) -> Dict[str, Pipeline]:
        """Load models from disk.

        Parameters
        ----------
        path : str
            Directory path containing saved models.

        Returns
        -------
        dict
            Mapping of model name to loaded Pipeline.
        """
        import os
        import joblib

        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")

        for filename in os.listdir(path):
            if filename.endswith(".joblib"):
                name = filename[:-7]  # strip .joblib
                filepath = os.path.join(path, filename)
                self.models[name] = joblib.load(filepath)
                logger.info("Loaded %s from %s", name, filepath)

        return self.models
