# lazypredict/metrics/anomaly_detection_metrics.py

from typing import Any, Dict, List, Optional, Tuple, Union

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
from sklearn import metrics

from .base import Metrics


class AnomalyDetectionMetrics(Metrics):
    """
    Calculates evaluation metrics for anomaly detection models.

    If true labels are provided, supervised metrics can be calculated.
    """

    def evaluate(self, y_true, y_pred) -> Dict[str, Any]:
        """
        Evaluates anomaly detection performance.

        Args:
            y_true: True labels (1 for normal, -1 for anomaly). Can be None for unsupervised metrics.
            y_pred: Predicted labels (1 for normal, -1 for anomaly).

        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        scores = {}

        # If y_true is provided, calculate supervised metrics
        if y_true is not None:
            accuracy = metrics.accuracy_score(y_true, y_pred)
            precision = metrics.precision_score(y_true, y_pred, pos_label=-1)
            recall = metrics.recall_score(y_true, y_pred, pos_label=-1)
            f1 = metrics.f1_score(y_true, y_pred, pos_label=-1)
            roc_auc = metrics.roc_auc_score(y_true, y_pred)

            scores['Accuracy'] = accuracy
            scores['Precision'] = precision
            scores['Recall'] = recall
            scores['F1 Score'] = f1
            scores['ROC AUC'] = roc_auc
        else:
            # Without y_true, unsupervised metrics are limited
            scores['Anomalies Detected'] = (y_pred == -1).sum()

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
        # Sort based on an appropriate metric, e.g., F1 Score
        if 'F1 Score' in df.columns:
            df.sort_values(by='F1 Score', ascending=False, inplace=True)
        elif 'Anomalies Detected' in df.columns:
            df.sort_values(by='Anomalies Detected', ascending=False, inplace=True)
        return df
