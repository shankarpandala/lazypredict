from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base import BaseEstimator

class LazyCausalAnalyzer(BaseEstimator):
    """
    LazyCausalAnalyzer for causal inference analysis using regression-based approaches.

    This class uses scikit-learn models to establish causal relationships in observational data
    by applying simple regression analysis.

    Attributes
    ----------
    model : object
        The causal analysis model used for establishing relationships.

    Methods
    -------
    fit(X, y):
        Trains the causal model on the provided data.
    predict(X):
        Generates predictions for the causal inference model.
    evaluate(X, y):
        Evaluates the model's performance using regression metrics.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : object, optional
            A scikit-learn compatible regression model. Defaults to LinearRegression.
        """
        self.model = model if model is not None else LinearRegression()

    def fit(self, X, y):
        """Train the causal analyzer on the provided data."""
        self.model.fit(X, y)

    def predict(self, X):
        """Generate causal predictions for the given input data."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the causal analyzer's performance using MSE, MAE, and R2 score."""
        predictions = self.predict(X)
        return {
            "mean_squared_error": mean_squared_error(y, predictions),
            "mean_absolute_error": mean_absolute_error(y, predictions),
            "r2_score": r2_score(y, predictions)
        }
