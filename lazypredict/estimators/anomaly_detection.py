from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from .base import BaseEstimator

class LazyAnomalyDetector(BaseEstimator):
    """
    LazyAnomalyDetector for automated training, prediction, and evaluation of anomaly detection models.

    This class uses scikit-learn anomaly detection models by default. It provides evaluation using metrics
    specific to binary classification on anomalous data.

    Attributes
    ----------
    model : object
        The anomaly detection model used for training and prediction.

    Methods
    -------
    fit(X):
        Trains the anomaly detection model on the provided data.
    predict(X):
        Generates anomaly predictions for the given input data.
    evaluate(X, y):
        Evaluates the model's performance using classification metrics for anomaly detection.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : object, optional
            A scikit-learn compatible anomaly detection model. Defaults to IsolationForest.
        """
        self.model = model if model is not None else IsolationForest()

    def fit(self, X, y=None):
        """Train the anomaly detector on the provided data."""
        self.model.fit(X)

    def predict(self, X):
        """Generate anomaly predictions for the given input data (1: normal, -1: anomaly)."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the anomaly detector's performance using F1, accuracy, precision, and recall."""
        predictions = self.predict(X)
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, pos_label=-1),
            "recall": recall_score(y, predictions, pos_label=-1),
            "f1_score": f1_score(y, predictions, pos_label=-1)
        }
