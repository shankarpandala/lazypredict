# lazypredict/metrics/anomaly_detection_metrics.py

import pandas as pd  # Added import for pandas
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, Dict, List
import logging
from .base import Metrics

logger = logging.getLogger(__name__)

class AnomalyDetectionMetrics(Metrics):
    """
    A class for calculating various anomaly detection metrics.
    """

    def calculate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Calculate anomaly detection metrics for the given true and predicted values.

        Args:
            y_true (Any): True target values.
            y_pred (Any): Predicted target values.

        Returns:
            Dict[str, float]: A dictionary of calculated metrics.
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate anomaly detection metrics: {e}")
            return {}

    def create_results_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of anomaly detection results from a list of dictionaries.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries containing results.

        Returns:
            pd.DataFrame: A DataFrame containing the anomaly detection results.
        """
        try:
            results_df = pd.DataFrame(results)
            return results_df
        except Exception as e:
            logger.error(f"Failed to create results DataFrame: {e}")
            return pd.DataFrame()
