# lazypredict/metrics/time_series_metrics.py

from typing import Any, Dict, List, Optional, Tuple, Union

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base import Metrics


class TimeSeriesMetrics(Metrics):
    """
    Calculates evaluation metrics for time series forecasting models.
    """

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates forecasting performance.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        scores = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2,
            'MAPE': mape,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = self.custom_metric(y_true, y_pred)

        return scores

    def create_results_df(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a DataFrame from the list of result dictionaries.

        Args:
            results (List[Dict[str, Any]]): List of evaluation results.

        Returns:
            pd.DataFrame: DataFrame of evaluation metrics.
        """
        df = pd.DataFrame(results)
        df.set_index('Model', inplace=True)
        df.sort_values(by='RMSE', ascending=True, inplace=True)
        return df
