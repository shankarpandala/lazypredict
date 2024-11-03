from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from .base import BaseEstimator

class LazyClusterer(BaseEstimator):
    """
    LazyClusterer for automated training, prediction, and evaluation of clustering models.

    This class uses scikit-learn clustering models by default. It provides training and evaluation
    metrics specific to clustering.

    Attributes
    ----------
    model : object
        The clustering model used for training and prediction.

    Methods
    -------
    fit(X):
        Trains the clustering model on the provided data.
    predict(X):
        Generates cluster predictions for the given input data.
    evaluate(X):
        Evaluates the model's performance using clustering metrics.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : object, optional
            A scikit-learn compatible clustering model. Defaults to KMeans.
        """
        self.model = model if model is not None else KMeans()

    def fit(self, X, y=None):
        """Train the clustering model on the provided data."""
        self.model.fit(X)
        self.labels = self.model.labels_

    def predict(self, X):
        """Generate cluster predictions for the given input data."""
        return self.model.predict(X)

    def evaluate(self, X):
        """Evaluate the clustering model using silhouette score, calinski_harabasz score, and davies_bouldin score."""
        labels = self.predict(X)
        return {
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels)
        }
