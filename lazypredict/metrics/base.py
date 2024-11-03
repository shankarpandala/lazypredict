# lazypredict/metrics/base.py

import pandas as pd  # Added import for pandas
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class Metrics(ABC):
    """
    Abstract base class for defining different metrics used for evaluation.
    """

    @abstractmethod
    def calculate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Calculate metrics for the given true and predicted values.

        Args:
            y_true (Any): True target values.
            y_pred (Any): Predicted target values.

        Returns:
            Dict[str, float]: A dictionary of calculated metrics.
        """
        pass

    def create_results_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame of results from a list of dictionaries.

        Args:
            results (List[Dict[str, Any]]): A list of dictionaries containing results.

        Returns:
            pd.DataFrame: A DataFrame containing the results.
        """
        try:
            results_df = pd.DataFrame(results)
            return results_df
        except Exception as e:
            logger.error(f"Failed to create results DataFrame: {e}")
            return pd.DataFrame()

