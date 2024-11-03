from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .base import MetricsBase

class ClusteringMetrics(MetricsBase):
    """
    ClusteringMetrics for evaluating clustering models.

    This class computes metrics such as silhouette score, calinski_harabasz score, and davies_bouldin score.

    Methods
    -------
    compute(X, labels):
        Compute clustering metrics.
    """

    def compute(self, X, labels):
        """
        Compute clustering metrics.

        Parameters
        ----------
        X : DataFrame
            Input data.
        labels : array-like
            Cluster labels.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        return {
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels)
        }
