from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from .base import BaseEstimator

class LazyRegressor(BaseEstimator):
    """
    LazyRegressor for automated training, prediction, and evaluation of regression models.

    This class uses scikit-learn regressors by default. It splits the provided data into training and test sets,
    trains the model, and provides predictions and evaluations with various metrics.

    Attributes
    ----------
    model : object
        The regression model used for training and prediction.
    
    Methods
    -------
    fit(X, y):
        Trains the regression model on the provided data.
    predict(X):
        Generates predictions for the given input data.
    evaluate(X, y):
        Evaluates the model's MSE, MAE, and R2 score.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : object, optional
            A scikit-learn compatible regressor model. Defaults to LinearRegression.
        """
        self.model = model if model is not None else LinearRegression()

    def fit(self, X, y):
        """Train the regressor on the provided data."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.X_test, self.y_test = X_test, y_test

    def predict(self, X):
        """Generate predictions for the given input data."""
        return self.model.predict(X)

    def evaluate(self, X=None, y=None):
        """Evaluate the regressor on MSE, MAE, and R2 score."""
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        predictions = self.predict(X)
        
        return {
            "mean_squared_error": mean_squared_error(y, predictions),
            "mean_absolute_error": mean_absolute_error(y, predictions),
            "r2_score": r2_score(y, predictions)
        }
