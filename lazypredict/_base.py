"""Base class for LazyClassifier and LazyRegressor — shared logic lives here."""

import difflib
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from lazypredict.config import (
    VALID_ENCODERS,
    get_cuml_models,
    get_gpu_model_params,
    is_gpu_available,
)
from lazypredict._timeout import time_limit
from lazypredict.exceptions import ModelFitError, TimeoutException  # noqa: F401
from lazypredict.integrations.mlflow import (
    MLFLOW_AVAILABLE,
    RunTracker,
    log_sklearn_model,
    setup_mlflow,
)
from lazypredict.metrics import metric_name
from lazypredict.preprocessing import (
    build_preprocessor,
    prepare_dataframes,
    with_string_column_names,
)
from lazypredict._optional import warn_if_broken

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
except Exception as exc:  # missing, or installed but broken
    _mlflow = None  # type: ignore[assignment]
    warn_if_broken("mlflow", exc)


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


def _resolve_estimators(
    selection: Union[str, List[Any]],
    available: List[Tuple[str, Any]],
    kind: str,
    allow_classes: bool = True,
) -> List[Tuple[str, Any]]:
    """Resolve a user's model selection into ``(name, class)`` pairs.

    Parameters
    ----------
    selection : "all", str or list
        ``"all"``, a single model name, or a list mixing model names and
        estimator classes.
    available : list of (str, type)
        Models that can be selected by name.
    kind : str
        Singular noun used in error messages, e.g. ``"classifier"``.
    allow_classes : bool, optional (default=True)
        Whether estimator classes may be passed besides names.

    Raises
    ------
    ValueError
        If a name is unknown or an element is neither a name nor a class.
    """
    if isinstance(selection, str):
        if selection == "all":
            return list(available)
        selection = [selection]

    by_name = dict(available)
    resolved: Dict[str, Any] = {}
    unknown: List[str] = []
    for item in selection:
        if isinstance(item, str):
            if item in by_name:
                resolved[item] = by_name[item]
            else:
                unknown.append(item)
        elif allow_classes and isinstance(item, type):
            resolved[item.__name__] = item
        elif allow_classes:
            raise ValueError(
                f"Invalid {kind} {item!r}: pass estimator classes or their names, "
                "not instances."
            )
        else:
            raise ValueError(f"Invalid {kind} {item!r}: pass {kind} names.")

    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, list(by_name), n=1)
            hints.append(f"{name!r} (did you mean {close[0]!r}?)" if close else repr(name))
        raise ValueError(
            f"Unknown {kind} name(s): {', '.join(hints)}. Names are case-sensitive; "
            "models from optional packages (xgboost, lightgbm, catboost, ...) are "
            f"only available when those packages are installed. Available: {sorted(by_name)}"
        )
    return list(resolved.items())


def _auto_convert_inputs(*args):
    """Auto-convert PySpark/Dask inputs to pandas/numpy."""
    from lazypredict.distributed import auto_convert_dataframe

    return tuple(auto_convert_dataframe(a, f"arg_{i}") for i, a in enumerate(args))


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
        # Tuning parameters (Phase 2)
        tune: bool = False,
        tune_top_k: int = 5,
        tune_trials: int = 50,
        tune_timeout: Optional[Union[int, float]] = None,
        tune_backend: str = "optuna",
    ):
        _validate_init_params(cv, timeout, categorical_encoder, custom_metric, n_jobs)
        if max_models is not None and (not isinstance(max_models, int) or max_models < 1):
            raise ValueError(f"max_models must be a positive integer, got {max_models!r}")
        for option, value in (("tune_top_k", tune_top_k), ("tune_trials", tune_trials)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{option} must be a positive integer, got {value!r}")
        if tune_backend not in ("optuna", "sklearn", "flaml"):
            raise ValueError(f"tune_backend must be 'optuna', 'sklearn', or 'flaml', got {tune_backend!r}")
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
        self.tune = tune
        self.tune_top_k = tune_top_k
        self.tune_trials = tune_trials
        self.tune_timeout = tune_timeout
        self.tune_backend = tune_backend
        self.tuned_scores_: Optional[Any] = None
        self.tuned_params_: Dict[str, Dict[str, Any]] = {}
        self.tuned_models_: Dict[str, Pipeline] = {}
        self._feature_columns: Optional[List[Any]] = None
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

    def _validate_target(self, y_train: Any) -> None:
        """Reject targets the estimator type cannot handle (optional hook)."""

    def _compute_metrics(
        self, pipe: Pipeline, X_test: pd.DataFrame, y_test: Any, X_train: pd.DataFrame
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_scores_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        raise NotImplementedError

    def _cv_scoring(self, y_train: Any, pipe: Pipeline) -> Optional[Dict[str, str]]:
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
            Prediction rows preserve the index of ``X_test``.
        """
        # Results of a previous fit() must not leak into this one.
        self.models = {}
        self.errors = {}
        self.tuned_scores_ = None
        self.tuned_params_ = {}
        self.tuned_models_ = {}

        # Auto-convert PySpark/Dask DataFrames
        X_train, X_test, y_train, y_test = _auto_convert_inputs(
            X_train, X_test, y_train, y_test
        )

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
        self._validate_target(y_train)
        X_train, X_test = prepare_dataframes(X_train, X_test)
        self._feature_columns = list(X_train.columns)
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
        tracker = RunTracker(_mlflow if self.mlflow_enabled and MLFLOW_AVAILABLE else None)
        with warnings.catch_warnings():
            if self.ignore_warnings:
                warnings.simplefilter("ignore")
            for idx, (name, model) in enumerate(
                progress_bar(estimator_list, disable=(self.verbose == 0))
            ):
                start = time.perf_counter()
                run_started = tracker.start(f"{self.__class__.__name__}-{name}", model_name=name)
                metrics: Optional[Dict[str, Any]] = None
                try:
                    model_kwargs = self._model_kwargs(model)
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", clone(preprocessor)),
                            (step_name, model(**model_kwargs)),
                        ]
                    )

                    with time_limit(self.timeout):
                        pipe.fit(X_train, y_train)
                    fit_time = time.perf_counter() - start
                    if self.timeout and fit_time > self.timeout:
                        # The limit could not be enforced preemptively
                        # (Windows, worker thread, or a long C call).
                        raise TimeoutException(
                            f"Training took {fit_time:.2f}s, exceeding timeout={self.timeout}s"
                        )

                    cv_metrics = self._run_cv(pipe, X_train, y_train, name) if self.cv else {}
                    metrics = self._compute_metrics(pipe, X_test, y_test, X_train)
                    metrics["name"] = name
                    metrics["time"] = time.perf_counter() - start
                    metrics.update(cv_metrics)
                    if self.custom_metric is not None:
                        metrics["custom_metric"] = self._custom_metric_value(
                            y_test, pipe.predict(X_test), name
                        )
                    if self.predictions:
                        predictions_dict[name] = pipe.predict(X_test)
                except Exception as exc:
                    metrics = None
                    self.errors[name] = exc
                    if isinstance(exc, TimeoutException):
                        logger.info("%s skipped: %s", name, exc)
                    elif not self.ignore_warnings:
                        logger.warning("%s model failed to execute: %s", name, exc)
                else:
                    # Only models that produced scores are kept.
                    self.models[name] = pipe
                    results.append(metrics)
                    if self.verbose > 0:
                        self._log_verbose(name, metrics)
                    if run_started:
                        tracker.call(self._log_mlflow_metrics, metrics, pipe, X_train, name)
                finally:
                    if run_started:
                        tracker.end()

                if self.progress_callback is not None:
                    self.progress_callback(name, idx + 1, total, metrics)

        scores = self._build_scores_dataframe(results)

        # Store for tuning access
        self._last_X_train = X_train
        self._last_y_train = y_train
        self._last_preprocessor = preprocessor
        self._last_estimator_list = estimator_list

        # Phase 2: Auto-tune top-k if requested
        if self.tune and len(scores) > 0:
            self.tuned_scores_ = self._run_tuning(scores, X_train, y_train, preprocessor, estimator_list)

        if self.predictions:
            prediction_index = X_test.index if isinstance(X_test, pd.DataFrame) else None
            predictions_df = pd.DataFrame(
                predictions_dict, index=prediction_index if predictions_dict else None
            )
            return scores, predictions_df
        return scores, pd.DataFrame()

    # -- Tuning integration ---------------------------------------------------

    def _tune_scoring(self) -> str:
        """Return the scoring metric string for tuning. Override in subclasses."""
        raise NotImplementedError

    def _run_tuning(
        self,
        scores: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: Any,
        preprocessor: Any,
        estimator_list: List,
    ) -> pd.DataFrame:
        """Tune the top-k models and refit them with their best parameters.

        Fills ``tuned_params_`` and ``tuned_models_`` (pipelines refitted on
        the training data) and returns the ``tuned_scores_`` table.
        """
        from lazypredict.tuning import tune_top_k

        scoring = self._tune_scoring()
        logger.info(
            "Tuning top %d models with %s backend (%d trials each)...",
            self.tune_top_k, self.tune_backend, self.tune_trials,
        )
        tuned = tune_top_k(
            scores_df=scores,
            models=self.models,
            estimator_list=estimator_list,
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            scoring=scoring,
            top_k=self.tune_top_k,
            n_trials=self.tune_trials,
            timeout=self.tune_timeout,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            backend=self.tune_backend,
            use_gpu=self.use_gpu,
            cv=self.cv or 5,
        )

        name_to_class = dict(estimator_list)
        step_name = self._estimator_step_name()
        for name, row in tuned.iterrows():
            if row["Status"] != "tuned":
                continue
            params = dict(row["Best Params"])
            model_class = name_to_class[name]
            try:
                pipe = Pipeline(steps=[
                    ("preprocessor", clone(preprocessor)),
                    (step_name, model_class(**{**self._model_kwargs(model_class), **params})),
                ])
                with warnings.catch_warnings():
                    if self.ignore_warnings:
                        warnings.simplefilter("ignore")
                    pipe.fit(X_train, y_train)
            except Exception as exc:
                logger.warning("Refitting tuned %s failed: %s", name, exc)
                continue
            self.tuned_params_[name] = params
            self.tuned_models_[name] = pipe
        return tuned

    def _model_kwargs(self, model_class: Any) -> Dict[str, Any]:
        """Seed, GPU and verbosity keyword arguments for a model class."""
        model_kwargs = get_gpu_model_params(model_class, self.use_gpu)
        if "random_state" in model_class().get_params():
            model_kwargs["random_state"] = self.random_state
        # Suppress verbose output for boosting libraries by default
        module = getattr(model_class, "__module__", "") or ""
        if "catboost" in module:
            model_kwargs.setdefault("verbose", 0)
        if "lightgbm" in module:
            model_kwargs.setdefault("verbose", -1)
            model_kwargs.setdefault("verbosity", -1)
        if "xgboost" in module:
            model_kwargs.setdefault("verbosity", 0)
        return model_kwargs

    # -- Explainability -------------------------------------------------------

    def explain(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        method: str = "permutation",
        n_repeats: int = 10,
        max_samples: int = 100,
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """Explain model predictions with feature importance.

        Parameters
        ----------
        X_test : array-like
            Test feature matrix.
        y_test : array-like
            Test target.
        method : str, optional (default='permutation')
            Explanation method: 'permutation' (zero deps) or 'shap'.
        n_repeats : int, optional (default=10)
            Repeats for permutation importance.
        max_samples : int, optional (default=100)
            Max samples for SHAP explanations.

        Returns
        -------
        pd.DataFrame or dict
            For 'permutation': DataFrame with features as rows, models as columns.
            For 'shap': dict mapping model name to SHAP values array.
        """
        if len(self.models) == 0:
            raise ValueError("No models fitted. Call fit() first.")

        X_test, y_test = _auto_convert_inputs(X_test, y_test)
        X_test = self._to_frame(X_test)

        if method == "permutation":
            from lazypredict.explainability import explain_permutation

            return explain_permutation(
                self.models, X_test, y_test,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        elif method == "shap":
            from lazypredict.explainability import explain_shap

            return explain_shap(
                self.models, X_test,
                step_name=self._estimator_step_name(),
                max_samples=max_samples,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'permutation' or 'shap'.")

    # -- Cross-validation -----------------------------------------------------

    def _run_cv(
        self, pipe: Pipeline, X_train: pd.DataFrame, y_train: Any, name: str
    ) -> Dict[str, Optional[float]]:
        """Run cross-validation and return metric dict."""
        from sklearn.model_selection import cross_validate

        scoring = self._cv_scoring(y_train, pipe)
        if scoring is None:
            return {}
        try:
            # error_score=nan: a scorer that cannot be computed for this model
            # (e.g. ROC AUC without predict_proba) only blanks its own column.
            cv_results = cross_validate(
                pipe, X_train, y_train,
                cv=self.cv, scoring=scoring,
                n_jobs=self.n_jobs, error_score=np.nan,
            )
            return self._process_cv_results(cv_results, X_train)
        except Exception as cv_exc:
            if not self.ignore_warnings:
                logger.warning("Cross-validation failed for %s: %s", name, cv_exc)
            return {key: np.nan for key in self._cv_column_names()}

    def _cv_column_names(self) -> List[str]:
        """Return the CV column names for null-filling on failure."""
        raise NotImplementedError

    def _log_verbose(self, name: str, metrics: Dict[str, Any]) -> None:
        """Log per-model metrics when verbose > 0."""
        raise NotImplementedError

    def _custom_metric_value(self, y_test: Any, y_pred: Any, name: str) -> Any:
        """The custom metric for one model, or NaN if it fails."""
        try:
            return self.custom_metric(y_test, y_pred)  # type: ignore[misc]
        except Exception as exc:
            if not self.ignore_warnings:
                logger.warning(
                    "Custom metric %s failed for %s: %s", metric_name(self.custom_metric), name, exc
                )
            return np.nan

    def _log_mlflow_metrics(
        self, metrics: Dict[str, Any], pipe: Pipeline, X_train: pd.DataFrame, name: str
    ) -> None:
        """Log the estimator's hyperparameters, the metrics and the model to MLflow."""
        if _mlflow is None:
            return
        estimator = pipe.steps[-1][1]
        _mlflow.log_params(
            {key: str(value)[:500] for key, value in estimator.get_params(deep=False).items()}
        )
        for key, val in metrics.items():
            if isinstance(val, (int, float, np.number)) and np.isfinite(val):
                _mlflow.log_metric(key, val)
        _mlflow.log_metric("training_time", metrics.get("time", 0))
        try:
            signature = _mlflow.models.infer_signature(X_train, pipe.predict(X_train))
            log_sklearn_model(_mlflow, pipe, f"{name}_model", signature=signature)
        except Exception as mlflow_exc:
            if not self.ignore_warnings:
                logger.warning("Failed to log model %s to MLflow: %s", name, mlflow_exc)

    def _to_frame(self, X: Any) -> pd.DataFrame:
        """Convert array-like input to a DataFrame with the training column names."""
        if isinstance(X, pd.DataFrame):
            return with_string_column_names(X)
        X = np.asarray(X)
        columns = self._feature_columns
        if columns is None and self.models:
            # Models loaded with load_models(): ask the fitted pipeline.
            first = next(iter(self.models.values()))
            columns = getattr(first, "feature_names_in_", None)
        if columns is None or len(columns) != X.shape[1]:
            columns = [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns).infer_objects()

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

        X_test = self._to_frame(X_test)

        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available models: {list(self.models.keys())}"
                )
            return np.asarray(self.models[model_name].predict(X_test))

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
