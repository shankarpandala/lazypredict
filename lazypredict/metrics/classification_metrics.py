# lazypredict/metrics/classification_metrics.py

import pandas as pd  # Added import for pandas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from typing import Any, Dict, List
import logging
from .base import Metrics

logger = logging.getLogger(__name__)

class ClassificationMetrics(Metrics):
    """
    A class for calculating various classification metrics.
    """

    def calculate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Calculate classification metrics for the given true and predicted values.

        Args:
            y_true (Any): True target values.
            y_pred (Any): Predicted target values.

        Returns:
            Dict[str, float]: A dictionary of calculated metrics.
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred, average='weighted'),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted')
            }

            # Calculate ROC AUC only if binary or probability predictions are available
            if len(set(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate classification metrics: {e}")
            return {}

    def create_results_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of classification results from a list of dictionaries.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries containing results.

        Returns:
            pd.DataFrame: A DataFrame containing the classification results.
        """
        try:
            results_df = pd.DataFrame(results)
            return results_df
        except Exception as e:
            logger.error(f"Failed to create results DataFrame: {e}")
            return pd.DataFrame()
