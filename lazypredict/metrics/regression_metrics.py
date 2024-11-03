# lazypredict/metrics/regression_metrics.py

import pandas as pd  # Added import for pandas
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Any, Dict, List
import logging
from .base import Metrics

logger = logging.getLogger(__name__)

class RegressionMetrics(Metrics):
    """
    A class for calculating various regression metrics.
    """

    def calculate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Calculate regression metrics for the given true and predicted values.

        Args:
            y_true (Any): True target values.
            y_pred (Any): Predicted target values.

        Returns:
            Dict[str, float]: A dictionary of calculated metrics.
        """
        try:
            metrics = {
                'mean_squared_error': mean_squared_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred),
                'mean_absolute_error': mean_absolute_error(y_true, y_pred)
            }
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate regression metrics: {e}")
            return {}

    def create_results_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of regression results from a list of dictionaries.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries containing results.

        Returns:
            pd.DataFrame: A DataFrame containing the regression results.
        """
        try:
            results_df = pd.DataFrame(results)
            return results_df
        except Exception as e:
            logger.error(f"Failed to create results DataFrame: {e}")
            return pd.DataFrame()
