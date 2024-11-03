# lazypredict/estimators/clustering.py

from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.base import ClusterMixin
from sklearn.utils import all_estimators
from tqdm.auto import tqdm

from .base import LazyEstimator
from ..metrics.clustering_metrics import ClusteringMetrics
from ..preprocessing.base import Preprocessor
from ..utils.backend import DataFrame
from ..utils.logging import get_logger
from ..utils.decorators import profile
from ..mlflow_integration.mlflow_logger import MLflowLogger

logger = get_logger(__name__)

CLUSTERER_EXCLUDES = [
    "AffinityPropagation",
    "AgglomerativeClustering",
    "Birch",
    "FeatureAgglomeration",
    "MiniBatchKMeans",
    "MeanShift",
    "SpectralClustering",
    "SpectralBiclustering",
    "SpectralCoclustering",
    # Add any other clusterers to exclude here
]


class LazyClusterer(LazyEstimator):
    """
    LazyClusterer is a class for automatically fitting and evaluating multiple
    clustering models with minimal code. It extends the LazyEstimator base
    class and implements the specific logic for clustering tasks.

    Attributes:
        metrics (ClusteringMetrics): Metrics object for clustering evaluation.
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
        metrics: Union[None, ClusteringMetrics] = None,
        profiling: bool = False,
        use_gpu: bool = False,
        mlflow_logging: bool = False,
        explainability: bool = False,
        n_clusters: int = 8,
    ):
        """
        Initializes the LazyClusterer with the given parameters.

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
            metrics (ClusteringMetrics, optional): Metrics object. Defaults to None.
            profiling (bool, optional): Enable profiling if True. Defaults to False.
            use_gpu (bool, optional): Enable GPU acceleration if True. Defaults to False.
            mlflow_logging (bool, optional): Enable MLflow logging if True. Defaults to False.
            explainability (bool, optional): Enable explainability features if True.
                Defaults to False.
            n_clusters (int, optional): Number of clusters to use. Defaults to 8.
        """
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            estimator_list=estimator_list,
            preprocessor=preprocessor,
            metrics=metrics or ClusteringMetrics(custom_metric),
            profiling=profiling,
            use_gpu=use_gpu,
            mlflow_logging=mlflow_logging,
            explainability=explainability,
        )
        self.n_clusters = n_clusters
        self.logger = logger

    def _get_estimators(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of clustering estimator classes to be used.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing estimator names and classes.
        """
        if self.estimator_list == "all":
            estimators = [
                (name, cls)
                for name, cls in all_estimators(type_filter="cluster")
                if issubclass(cls, ClusterMixin) and name not in CLUSTERER_EXCLUDES
            ]
            # Include any GPU-compatible clusterers if use_gpu is True
            if self.use_gpu:
                estimators.extend(self._get_gpu_clusterers())
            return estimators
        else:
            return [(est.__name__, est) for est in self.estimator_list]

    def _get_gpu_clusterers(self) -> List[Tuple[str, Any]]:
        """
        Retrieves a list of GPU-compatible clusterers.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing clusterer names and classes.
        """
        gpu_estimators = []
        try:
            from cuml import KMeans, DBSCAN, GaussianMixture

            gpu_estimators = [
                ("KMeans", KMeans),
                ("DBSCAN", DBSCAN),
                ("GaussianMixture", GaussianMixture),
                # Add other cuML clusterers here
            ]
        except ImportError:
            self.logger.warning("cuML is not installed. GPU clusterers are unavailable.")
        return gpu_estimators

    @profile(profiling_attr='profiling')
    def fit(
        self,
        X: DataFrame,
        y: Union[None, DataFrame] = None,
    ) -> Union[Tuple[DataFrame, Dict[str, Any]], DataFrame]:
        """
        Fits multiple clustering models and evaluates them.

        Args:
            X (DataFrame): Feature set.
            y (DataFrame, optional): True labels for evaluation (if available).
                Defaults to None.

        Returns:
            Union[Tuple[DataFrame, Dict[str, Any]], DataFrame]: DataFrame of evaluation
            results and optionally a dictionary of cluster assignments.
        """
        X = self._prepare_data(X)
        if y is not None:
            y = self._prepare_data(y)

        self.preprocessor.fit(X)

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
                X=X,
                y=y,
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
        X: DataFrame,
        y: Union[None, DataFrame],
    ) -> Union[Dict[str, Any], None]:
        """
        Fits a single clusterer and evaluates it.

        Args:
            estimator_name (str): Name of the estimator.
            estimator_cls (Any): Estimator class.
            X (DataFrame): Feature set.
            y (DataFrame, optional): True labels for evaluation (if available).

        Returns:
            Union[Dict[str, Any], None]: Evaluation results or None if an error occurred.
        """
        start_time = time.time()
        try:
            estimator = self._initialize_estimator(estimator_cls)
            pipeline = self._create_pipeline([
                ('preprocessor', self.preprocessor.build_preprocessor()),
                ('estimator', estimator)
            ])
            pipeline.fit(X)

            cluster_labels = pipeline.predict(X)

            if self.predictions:
                self.predictions_dict[estimator_name] = cluster_labels

            scores = self.metrics.evaluate(X, cluster_labels, y)
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
                    pipeline, X, cluster_labels, estimator_name
                )

            self._log(f"Completed {estimator_name} in {elapsed_time:.4f} seconds.")

            return result
        except Exception as e:
            if not self.ignore_warnings:
                self.logger.exception(f"Model {estimator_name} failed to execute.")
            return None

    def _initialize_estimator(self, estimator_cls: Any) -> Any:
        """
        Initializes the estimator with appropriate parameters.

        Args:
            estimator_cls (Any): Estimator class.

        Returns:
            Any: Initialized estimator.
        """
        estimator_params = {}
        if "random_state" in estimator_cls().get_params():
            estimator_params["random_state"] = self.random_state
        if "n_clusters" in estimator_cls().get_params():
            estimator_params["n_clusters"] = self.n_clusters

        estimator = estimator_cls(**estimator_params)

        # If GPU is enabled, replace estimator with GPU version if available
        estimator = self._maybe_use_gpu_estimator(estimator)

        return estimator
