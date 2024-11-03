# lazypredict/estimators/causal_analysis.py

from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from .base import LazyEstimator
from ..metrics.causal_analysis_metrics import CausalAnalysisMetrics
from ..preprocessing.base import Preprocessor
from ..utils.backend import DataFrame
from ..utils.logging import get_logger
from ..utils.decorators import profile
from ..mlflow_integration.mlflow_logger import MLflowLogger

logger = get_logger(__name__)

CAUSAL_MODEL_EXCLUDES = [
    # List any causal models to exclude
]

# Import causal inference models
try:
    from dowhy import CausalModel
except ImportError:
    logger.warning("DoWhy library is not installed. Causal analysis models are unavailable.")

try:
    import econml
except ImportError:
    logger.warning("EconML library is not installed. Advanced causal analysis models are unavailable.")


class LazyCausalAnalyzer(LazyEstimator):
    """
    LazyCausalAnalyzer automatically fits and evaluates multiple causal analysis models.

    It extends the LazyEstimator base class and implements specific logic for causal analysis tasks.
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
        metrics: Union[None, CausalAnalysisMetrics] = None,
        profiling: bool = False,
        use_gpu: bool = False,
        mlflow_logging: bool = False,
        explainability: bool = False,
        treatment: str = None,
        outcome: str = None,
        common_causes: List[str] = None,
        effect_modifiers: List[str] = None,
    ):
        """
        Initializes the LazyCausalAnalyzer.

        Args:
            verbose (int, optional): Verbosity level. Defaults to 0.
            ignore_warnings (bool, optional): Suppress warnings if True. Defaults to True.
            custom_metric (callable, optional): Custom evaluation metric function. Defaults to None.
            predictions (bool, optional): Store predictions if True. Defaults to False.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            estimator_list (Union[str, List[Any]], optional): List of estimators or 'all'. Defaults to "all".
            preprocessor (Preprocessor, optional): Preprocessor object. Defaults to None.
            metrics (CausalAnalysisMetrics, optional): Metrics object. Defaults to None.
            profiling (bool, optional): Enable profiling if True. Defaults to False.
            use_gpu (bool, optional): Enable GPU acceleration if True. Defaults to False.
            mlflow_logging (bool, optional): Enable MLflow logging if True. Defaults to False.
            explainability (bool, optional): Enable explainability features if True. Defaults to False.
            treatment (str, optional): Name of the treatment variable. Required.
            outcome (str, optional): Name of the outcome variable. Required.
            common_causes (List[str], optional): List of common cause variables. Defaults to None.
            effect_modifiers (List[str], optional): List of effect modifier variables. Defaults to None.
        """
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            estimator_list=estimator_list,
            preprocessor=preprocessor,
            metrics=metrics or CausalAnalysisMetrics(custom_metric),
            profiling=profiling,
            use_gpu=use_gpu,
            mlflow_logging=mlflow_logging,
            explainability=explainability,
        )
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.effect_modifiers = effect_modifiers
        self.logger = logger

        if self.treatment is None or self.outcome is None:
            raise ValueError("Both 'treatment' and 'outcome' parameters must be specified.")

    def _get_estimators(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of causal analysis estimator classes to be used.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing estimator names and classes.
        """
        estimators = []
        if self.estimator_list == "all":
            estimators.extend(self._get_default_estimators())
            # Include any GPU-compatible causal models if use_gpu is True
            if self.use_gpu:
                estimators.extend(self._get_gpu_causal_models())
        else:
            estimators = [(est.__name__, est) for est in self.estimator_list]
        return estimators

    def _get_default_estimators(self) -> List[Tuple[str, Any]]:
        """
        Returns a list of default causal analysis models.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing estimator names and classes.
        """
        estimators = []
        try:
            from dowhy import CausalModel
            estimators.append(("DoWhy", CausalModel))
        except ImportError:
            self.logger.warning("DoWhy library is not installed.")

        try:
            from econml.dml import DML
            estimators.append(("DoubleML", DML))
        except ImportError:
            self.logger.warning("EconML library is not installed.")

        # Add more causal models here

        return estimators

    def _get_gpu_causal_models(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of GPU-compatible causal analysis models.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing model names and classes.
        """
        gpu_estimators = []
        # Currently, GPU support for causal models is limited
        # Add GPU-compatible causal models if available
        return gpu_estimators

    @profile(profiling_attr='profiling')
    def fit(
        self,
        data: DataFrame,
    ) -> Union[Tuple[DataFrame, Dict[str, Any]], DataFrame]:
        """
        Fits multiple causal analysis models and evaluates them.

        Args:
            data (DataFrame): DataFrame containing treatment, outcome, and other variables.

        Returns:
            Union[Tuple[DataFrame, Dict[str, Any]], DataFrame]: DataFrame of evaluation results and optionally predictions.
        """
        data = self._prepare_data(data)

        results = []
        estimators = self._get_estimators()

        estimator_iterator = tqdm(estimators) if self.verbose else estimators

        for name, estimator_cls in estimator_iterator:
            result = self._fit_and_evaluate(
                estimator_name=name,
                estimator_cls=estimator_cls,
                data=data,
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
        data: DataFrame,
    ) -> Union[Dict[str, Any], None]:
        """
        Fits a single causal analysis model and evaluates it.

        Args:
            estimator_name (str): Name of the estimator.
            estimator_cls (Any): Estimator class.
            data (DataFrame): DataFrame containing treatment, outcome, and other variables.

        Returns:
            Union[Dict[str, Any], None]: Evaluation results or None if an error occurred.
        """
        start_time = time.time()
        try:
            model = self._initialize_estimator(estimator_cls, data)
            estimate = self._estimate_effect(model)
            if estimate is None:
                return None

            if self.predictions:
                self.predictions_dict[estimator_name] = estimate.value

            scores = self.metrics.evaluate(estimate)
            elapsed_time = time.time() - start_time

            result = {
                'Model': estimator_name,
                **scores,
                'Time Taken': elapsed_time,
            }

            self.models[estimator_name] = model

            if self.mlflow_logging:
                self.mlflow_logger.log_model(model, estimator_name)
                self.mlflow_logger.log_metrics(scores)

            if self.explainability:
                self._generate_explainability(
                    model, data, estimate, estimator_name
                )

            self._log(f"Completed {estimator_name} in {elapsed_time:.4f} seconds.")

            return result
        except Exception as e:
            if not self.ignore_warnings:
                self.logger.exception(f"Model {estimator_name} failed to execute.")
            return None

    def _initialize_estimator(self, estimator_cls: Any, data: DataFrame) -> Any:
        """
        Initializes the causal model with appropriate parameters.

        Args:
            estimator_cls (Any): Estimator class.
            data (DataFrame): DataFrame containing treatment, outcome, and other variables.

        Returns:
            Any: Initialized causal model.
        """
        if estimator_cls.__name__ == "CausalModel":
            model = estimator_cls(
                data=data,
                treatment=self.treatment,
                outcome=self.outcome,
                common_causes=self.common_causes,
                effect_modifiers=self.effect_modifiers,
            )
        elif estimator_cls.__name__ == "DML":
            from sklearn.linear_model import Lasso
            model = estimator_cls(
                model_y=Lasso(),
                model_t=Lasso(),
                random_state=self.random_state,
            )
        else:
            raise NotImplementedError(f"Model {estimator_cls.__name__} is not implemented.")
        return model

    def _estimate_effect(self, model: Any) -> Any:
        """
        Estimates the causal effect using the model.

        Args:
            model (Any): Initialized causal model.

        Returns:
            Any: Estimated effect.
        """
        if model.__class__.__name__ == "CausalModel":
            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
            )
            return estimate
        elif model.__class__.__name__ == "DML":
            X = model._data.drop(columns=[self.treatment, self.outcome])
            T = model._data[self.treatment]
            Y = model._data[self.outcome]
            model.fit(Y, T, X=X)
            treatment_effect = model.effect(X)
            estimate = type('Estimate', (object,), {'value': treatment_effect.mean()})
            return estimate
        else:
            return None

    def _generate_explainability(self, model: Any, data: DataFrame, estimate: Any, model_name: str):
        """
        Generates explainability reports for the model.

        Args:
            model (Any): Trained causal model.
            data (DataFrame): DataFrame used for analysis.
            estimate (Any): Estimated causal effect.
            model_name (str): Name of the model.
        """
        # Causal analysis explainability may involve interpreting causal graphs and effects
        if model.__class__.__name__ == "CausalModel":
            model.view_model()
            model.view_estimate(estimate)
            # Additional explainability methods can be added here
        pass
