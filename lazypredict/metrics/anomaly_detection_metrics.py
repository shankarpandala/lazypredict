from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base import MetricsBase

class AnomalyDetectionMetrics(MetricsBase):
    """
    AnomalyDetectionMetrics for evaluating anomaly detection models.

    This class computes metrics such as accuracy, precision, recall, and F1 score.

    Methods
    -------
    compute(y_true, y_pred):
        Compute anomaly detection metrics.
    """

    def compute(self, y_true, y_pred):
        """
        Compute anomaly detection metrics.

        Parameters
        ----------
        y_true : array-like
            True labels (-1 for anomaly, 1 for normal).
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=-1),
            "recall": recall_score(y_true, y_pred, pos_label=-1),
            "f1_score": f1_score(y_true, y_pred, pos_label=-1)
        }
