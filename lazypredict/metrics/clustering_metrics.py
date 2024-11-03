# lazypredict/metrics/clustering_metrics.py

import pandas as pd  # Added import for pandas
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Any, Dict, List
import logging
from .base import Metrics

logger = logging.getLogger(__name__)

class ClusteringMetrics(Metrics):
    """
    A class for calculating various clustering metrics.
    """

    def calculate(self, X: Any, labels: Any) -> Dict[str, float]:
        """
        Calculate clustering metrics for the given data and cluster labels.

        Args:
            X (Any): Input features.
            labels (Any): Cluster labels.

        Returns:
            Dict[str, float]: A dictionary of calculated metrics.
        """
        try:
            metrics = {
                'silhouette_score': silhouette_score(X, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X, labels),
                'davies_bouldin_score': davies_bouldin_score(X, labels)
            }
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate clustering metrics: {e}")
            return {}

    def create_results_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of clustering results from a list of dictionaries.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries containing results.

        Returns:
            pd.DataFrame: A DataFrame containing the clustering results.
        """
        try:
            results_df = pd.DataFrame(results)
            return results_df
        except Exception as e:
            logger.error(f"Failed to create results DataFrame: {e}")
            return pd.DataFrame()
