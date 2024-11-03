from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """
    Abstract base class for all estimators in LazyPredict.

    This class provides a common interface for model training, prediction, and evaluation.
    Concrete classes must implement `fit`, `predict`, and `evaluate` methods.

    Methods
    -------
    fit(X, y):
        Trains the estimator on the provided data.
    predict(X):
        Makes predictions on the provided data.
    evaluate(X, y):
        Evaluates the estimator on the provided test data.
    """

    @abstractmethod
    def fit(self, X, y):
        """Train the estimator with the given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for the given input data."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate the estimator's performance on the test data."""
        pass
