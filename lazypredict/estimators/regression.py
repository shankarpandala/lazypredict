# lazypredict/estimators/regression.py

from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.base import RegressorMixin
from sklearn.utils import all_estimators
from tqdm.auto import tqdm

from .base import LazyEstimator
from ..metrics.regression_metrics import RegressionMetrics
from ..preprocessing.base import Preprocessor
from ..utils.backend import DataFrame, Series
from ..utils.logging import get_logger
from ..utils.decorators import profile
from ..mlflow_integration.mlflow_logger import MLflowLogger
from ..utils.backend import Backend

logger = get_logger(__name__)

REGRESSOR_EXCLUDES = [
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
    "QuantileRegressor",
    # Add any other regressors to exclude here
]


class LazyRegressor(LazyEstimator):
    """
    LazyRegressor is a class for automatically fitting and evaluating multiple
    regression models with minimal code. It extends the LazyEstimator base
    class and implements the specific logic for regression tasks.

    Attributes:
        metrics (RegressionMetrics): Metrics object for regression evaluation.
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
        metrics: Union[None, RegressionMetrics] = None,
        profiling: bool = False,
        use_gpu: bool = False,
        mlflow_logging: bool = False,
        explainability: bool = False,
    ):
        """
        Initializes the LazyRegressor with the given parameters.

        Args:
            verbose (int, optional): Verbosity level. Defaults to 0.
            ignore_warnings (bool, optional): Suppress warnings if True. Defaults to True.
            custom_metric (callable, optional): Custom evaluation metric function.
                Defaults to None.
            predictions (bool, optional): Store predictions if True. Defaults to False.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            estimator_list (Union[str, List[Any]], optional): List of estimators or 'all'.
                Defaults to "all".
            preprocessor (Preprocessor, optional): Preprocessor object. Defaults to None.
            metrics (RegressionMetrics, optional): Metrics object. Defaults to None.
            profiling (bool, optional): Enable profiling if True. Defaults to False.
            use_gpu (bool, optional): Enable GPU acceleration if True. Defaults to False.
            mlflow_logging (bool, optional): Enable MLflow logging if True. Defaults to False.
            explainability (bool, optional): Enable explainability features if True.
                Defaults to False.
        """
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            estimator_list=estimator_list,
            preprocessor=preprocessor,
            metrics=metrics or RegressionMetrics(custom_metric),
            profiling=profiling,
            use_gpu=use_gpu,
            mlflow_logging=mlflow_logging,
            explainability=explainability,
        )
        self.logger = logger

    def _get_estimators(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of regression estimator classes to be used.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing estimator names and classes.
        """
        if self.estimator_list == "all":
            estimators = [
                (name, cls)
                for name, cls in all_estimators(type_filter="regressor")
                if issubclass(cls, RegressorMixin) and name not in REGRESSOR_EXCLUDES
            ]
            # Include any GPU-compatible regressors if use_gpu is True
            if self.use_gpu:
                estimators.extend(self._get_gpu_regressors())
            return estimators
        else:
            return [(est.__name__, est) for est in self.estimator_list]

    def _get_gpu_regressors(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of GPU-compatible regressors.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing regressor names and classes.
        """
        gpu_estimators = []
        try:
            from cuml import linear_model, ensemble, svm

            gpu_estimators = [
                ("LinearRegression", linear_model.LinearRegression),
                ("Ridge", linear_model.Ridge),
                ("Lasso", linear_model.Lasso),
                ("ElasticNet", linear_model.ElasticNet),
                ("SVR", svm.SVR),
                ("RandomForestRegressor", ensemble.RandomForestRegressor),
                # Add other cuML regressors here
            ]
        except ImportError:
            self.logger.warning("cuML is not installed. GPU regressors are unavailable.")
        return gpu_estimators

    @profile(profiling_attr='profiling')
    def fit(
        self,
        X_train: DataFrame,
        X_test: DataFrame,
        y_train: Series,
        y_test: Series,
    ) -> Union[Tuple[DataFrame, Dict[str, Any]], DataFrame]:
        """
        Fits multiple regression models and evaluates them on test data.

        Args:
            X_train (DataFrame): Training feature set.
            X_test (DataFrame): Test feature set.
            y_train (Series): Training target values.
            y_test (Series): Test target values.

        Returns:
            Union[Tuple[DataFrame, Dict[str, Any]], DataFrame]: DataFrame of evaluation
            results and optionally a dictionary of predictions.
        """
        X_train = self._prepare_data(X_train)
        X_test = self._prepare_data(X_test)
        y_train = self._prepare_data(y_train)
        y_test = self._prepare_data(y_test)

        self.preprocessor.fit(X_train)

        results = []
        estimators = self._get_estimators()

        if self.verbose:
            estimator_iterator = tqdm(estimators)
        else:
            estimator_iterator = estimators

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
        X_train: DataFrame,
        y_train: Series,
        X_test: DataFrame,
        y_test: Series,
    ) -> Union[Dict[str, Any], None]:
        """
        Fits a single regressor and evaluates it.

        Args:
            estimator_name (str): Name of the estimator.
            estimator_cls (Any): Estimator class.
            X_train (DataFrame): Training feature set.
            y_train (Series): Training target values.
            X_test (DataFrame): Test feature set.
            y_test (Series): Test target values.

        Returns:
            Union[Dict[str, Any], None]: Evaluation results or None if an error occurred.
        """
        start_time = time.time()
        try:
            pipeline = self._build_pipeline(estimator_cls)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if self.predictions:
                self.predictions_dict[estimator_name] = y_pred

            scores = self.metrics.evaluate(y_test, y_pred, X_test.shape[1])
            elapsed_time = time.time() - start_time

            result = {
                'Model': estimator_name,
                **scores,
                'Time Taken': elapsed_time,
            }

            self.models[estimator_name] = pipeline

            if self.mlflow_logging:
                self.mlflow_logger.log_model(pipeline, estimator_name)
                self.mlflow_logger.log_metrics(scores)

            if self.explainability:
                self._generate_explainability(
                    pipeline, X_test, y_test, estimator_name
                )

            self._log(f"Completed {estimator_name} in {elapsed_time:.4f} seconds.")

            return result
        except Exception as e:
            if not self.ignore_warnings:
                self.logger.exception(f"Model {estimator_name} failed to execute.")
            return None
