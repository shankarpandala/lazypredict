from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base import MetricsBase

class ClassificationMetrics(MetricsBase):
    """
    ClassificationMetrics for evaluating classification models.

    This class computes metrics such as accuracy, precision, recall, and F1 score.

    Methods
    -------
    compute(y_true, y_pred):
        Compute accuracy, precision, recall, and F1 score.
    """

    def compute(self, y_true, y_pred):
        """
        Compute classification metrics.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted")
        }
