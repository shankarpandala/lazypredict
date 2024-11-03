from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base import MetricsBase

class CausalAnalysisMetrics(MetricsBase):
    """
    CausalAnalysisMetrics for evaluating causal analysis models.

    This class computes metrics such as MSE, MAE, and R2 score.

    Methods
    -------
    compute(y_true, y_pred):
        Compute causal analysis metrics.
    """

    def compute(self, y_true, y_pred):
        """
        Compute causal analysis metrics.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        return {
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred)
        }
