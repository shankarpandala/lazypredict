# lazypredict/estimators/classification.py

from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import time

from .base import LazyEstimator
from ..metrics.classification_metrics import ClassificationMetrics
from ..preprocessing.base import Preprocessor
from ..utils.backend import Backend
from ..utils.logging import get_logger
from ..utils.decorators import profile

logger = get_logger(__name__)

CLASSIFIER_EXCLUDES = [
    # List classifiers to exclude if necessary
]


class LazyClassifier(LazyEstimator):
    """
    LazyClassifier automatically fits and evaluates multiple classification models.
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Union[None, callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        estimator_list: Union[str, List[Any]] = "all",
        preprocessor: Union[None, Preprocessor] = None,
        metrics: Union[None, ClassificationMetrics] = None,
        profiling: bool = False,
        use_gpu: bool = False,
        mlflow_logging: bool = False,
        explainability: bool = False,
    ):
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            estimator_list=estimator_list,
            preprocessor=preprocessor,
            metrics=metrics or ClassificationMetrics(custom_metric),
            profiling=profiling,
            use_gpu=use_gpu,
            mlflow_logging=mlflow_logging,
            explainability=explainability,
        )
        self.backend = Backend.get_backend()
        self.logger = logger

    def _get_estimators(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of classification estimator classes to be used.
        """
        if self.estimator_list == "all":
            estimators = [
                (name, cls)
                for name, cls in all_estimators(type_filter='classifier')
                if issubclass(cls, ClassifierMixin) and name not in CLASSIFIER_EXCLUDES
            ]
            # Include any GPU-compatible classifiers if use_gpu is True
            if self.use_gpu:
                estimators.extend(self._get_gpu_classifiers())
            return estimators
        else:
            return [(est.__name__, est) for est in self.estimator_list]

    def _get_gpu_classifiers(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of GPU-compatible classifiers.
        """
        gpu_estimators = []
        try:
            from cuml import RandomForestClassifier, LogisticRegression

            gpu_estimators = [
                ("RandomForestClassifier", RandomForestClassifier),
                ("LogisticRegression", LogisticRegression),
                # Add other cuML classifiers here
            ]
        except ImportError:
            self.logger.warning("cuML is not installed. GPU classifiers are unavailable.")
        return gpu_estimators

    @profile(profiling_attr='profiling')
    def fit(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
    ) -> Union[Tuple[Any, Dict[str, Any]], Any]:
        """
        Fits multiple classification models and evaluates them.
        """
        X_train = self._prepare_data(X_train)
        X_test = self._prepare_data(X_test)
        y_train = self._prepare_data(y_train)
        y_test = self._prepare_data(y_test)

        self.preprocessor.fit(X_train, y_train)

        results = []
        estimators = self._get_estimators()

        estimator_iterator = tqdm(estimators) if self.verbose else estimators

        for name, estimator_cls in estimator_iterator:
            result = self._fit_and_evaluate(
                estimator_name=name,
                estimator_cls=estimator_cls,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            if result:
                results.append(result)

        results_df = self.metrics.create_results_df(results)
        if self.predictions:
            return results_df, self.predictions_dict
        else:
            return results_df

    def _fit_and_evaluate(
        self,
        estimator_name: str,
        estimator_cls: Any,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> Union[Dict[str, Any], None]:
        """
        Fits a single classifier and evaluates it.
        """
        start_time = time.time()
        try:
            estimator = self._initialize_estimator(estimator_cls)
            pipeline = self._create_pipeline([
                ('preprocessor', self.preprocessor.build_preprocessor()),
                ('estimator', estimator)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            if self.predictions:
                self.predictions_dict[estimator_name] = y_pred

            scores = self.metrics.evaluate(y_test, y_pred)
            elapsed_time = time.time() - start_time

            result = {
                'Model': estimator_name,
                **scores,
                'Time Taken': elapsed_time,
            }

            self.models[estimator_name] = pipeline

            if self.mlflow_logging:
                self.mlflow_logger.log_model(pipeline, artifact_path=estimator_name, X=X_test)
                self.mlflow_logger.log_metrics(scores)

            if self.explainability:
                self._generate_explainability(
                    model=pipeline,
                    X=X_test,
                    y=y_test,
                    model_name=estimator_name
                )

            self._log(f"Completed {estimator_name} in {elapsed_time:.4f} seconds.")

            return result
        except Exception as e:
            if not self.ignore_warnings:
                self.logger.exception(f"Model {estimator_name} failed to execute: {e}")
            return None

    def _initialize_estimator(self, estimator_cls: Any) -> Any:
        """
        Initializes the estimator with appropriate parameters.
        """
        estimator_params = {}
        if "random_state" in estimator_cls().get_params():
            estimator_params["random_state"] = self.random_state

        estimator = estimator_cls(**estimator_params)

        # If GPU is enabled, replace estimator with GPU version if available
        estimator = self._maybe_use_gpu_estimator(estimator)

        return estimator
