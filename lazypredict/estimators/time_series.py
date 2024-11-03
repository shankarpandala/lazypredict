from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import BaseEstimator

class LazyTimeSeriesForecaster(BaseEstimator):
    """
    LazyTimeSeriesForecaster for automated training, prediction, and evaluation of time series forecasting models.

    This class uses statsmodels time series models by default. It provides evaluation metrics specific
    to regression-like tasks for time series data.

    Attributes
    ----------
    model : object
        The time series forecasting model used for training and prediction.

    Methods
    -------
    fit(X, y):
        Trains the time series forecasting model on the provided data.
    predict(X):
        Generates time series predictions for the given input data.
    evaluate(X, y):
        Evaluates the model's performance using time series regression metrics.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : object, optional
            A statsmodels compatible time series model. Defaults to ExponentialSmoothing.
        """
        self.model = model

    def fit(self, X, y):
        """Train the time series forecaster on the provided data."""
        self.model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()

    def predict(self, steps=1):
        """Generate time series predictions for a specified number of steps ahead."""
        return self.model.forecast(steps)

    def evaluate(self, y_true, y_pred):
        """Evaluate the time series forecaster's performance using MSE, MAE, and R2 score."""
        return {
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred)
        }
