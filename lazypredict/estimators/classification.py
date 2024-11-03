from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from .base import BaseEstimator

class LazyClassifier(BaseEstimator):
    """
    LazyClassifier for automated training, prediction, and evaluation of classification models.

    This class uses scikit-learn classifiers by default. It splits the provided data into training and test sets,
    trains the model, and provides predictions and evaluations with various metrics.

    Attributes
    ----------
    model : object
        The classification model used for training and prediction.
    
    Methods
    -------
    fit(X, y):
        Trains the classification model on the provided data.
    predict(X):
        Generates predictions for the given input data.
    evaluate(X, y):
        Evaluates the model's accuracy, precision, recall, and F1 score.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : object, optional
            A scikit-learn compatible classifier model. Defaults to RandomForestClassifier.
        """
        self.model = model if model is not None else RandomForestClassifier()

    def fit(self, X, y):
        """Train the classifier on the provided data."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.X_test, self.y_test = X_test, y_test

    def predict(self, X):
        """Generate predictions for the given input data."""
        return self.model.predict(X)

    def evaluate(self, X=None, y=None):
        """Evaluate the classifier on accuracy, precision, recall, and F1 score."""
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        predictions = self.predict(X)
        
        return {
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(y, predictions, average='weighted'),
            "recall": recall_score(y, predictions, average='weighted'),
            "f1_score": f1_score(y, predictions, average='weighted')
        }
