# lazypredict/estimators/base.py

from typing import Any, Dict, List, Optional, Tuple, Union

import time
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from ..preprocessing.base import Preprocessor
from ..utils.backend import Backend
from ..utils.logging import get_logger
from ..utils.decorators import profile
from ..mlflow_integration.mlflow_logger import MLflowLogger
from ..explainability.shap_explainer import ShapExplainer

logger = get_logger(__name__)


class LazyEstimator:
    """
    Base class for lazy estimators.
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        estimator_list: Union[str, List[Any]] = "all",
        preprocessor: Optional[Preprocessor] = None,
        metrics: Any = None,
        profiling: bool = False,
        use_gpu: bool = False,
        mlflow_logging: bool = False,
        explainability: bool = False,
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.random_state = random_state
        self.estimator_list = estimator_list
        self.preprocessor = preprocessor or Preprocessor(use_gpu=use_gpu)
        self.metrics = metrics
        self.profiling = profiling
        self.use_gpu = use_gpu
        self.mlflow_logging = mlflow_logging
        self.explainability = explainability

        self.models = {}
        self.predictions_dict = {}
        self.backend = Backend.get_backend()
        self.logger = logger

        if self.mlflow_logging:
            self.mlflow_logger = MLflowLogger(
                experiment_name="LazyPredict Experiment",
                run_name=f"{self.__class__.__name__} Run",
                tags={"estimator_type": self.__class__.__name__},
            )

    def _prepare_data(self, X: Any) -> Any:
        """
        Prepares input data by converting it to the appropriate DataFrame type.
        """
        backend = self.backend
        DataFrame = backend.DataFrame

        if not isinstance(X, DataFrame):
            try:
                X = DataFrame(X)
            except Exception as e:
                self.logger.error(f"Failed to convert data to DataFrame: {e}")
                raise ValueError("Input data cannot be converted to a DataFrame.")
        return X

    def _create_pipeline(self, steps: List[Tuple[str, Any]]) -> Pipeline:
        """
        Creates a scikit-learn Pipeline with the provided steps.
        """
        return Pipeline(steps=steps)

    def _maybe_use_gpu_estimator(self, estimator: Any) -> Any:
        """
        Replaces the estimator with its GPU equivalent if available and use_gpu is True.
        """
        if self.use_gpu:
            gpu_estimator = self.backend.get_gpu_estimator(estimator)
            if gpu_estimator is not None:
                estimator = gpu_estimator
                self.logger.info(f"Using GPU estimator: {estimator.__class__.__name__}")
        return estimator

    def _log(self, message: str):
        """
        Logs a message if verbose is enabled.
        """
        if self.verbose:
            self.logger.info(message)

    def fit(self, *args, **kwargs):
        """
        Placeholder fit method to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _generate_explainability(self, model: Any, X: Any, y: Any, model_name: str):
        """
        Generates explainability reports using SHAP.
        """
        if self.explainability:
            try:
                shap_explainer = ShapExplainer(model, X, use_gpu=self.use_gpu)
                shap_values = shap_explainer.compute_shap_values()
                shap_explainer.plot_shap_summary(shap_values, model_name)
                self.logger.info(f"Generated explainability for {model_name}.")
            except Exception as e:
                self.logger.exception(f"Failed to generate explainability for {model_name}: {e}")

    def __del__(self):
        """
        Ensures that the MLflow run is ended when the estimator is destroyed.
        """
        if self.mlflow_logging:
            self.mlflow_logger.end_run()
