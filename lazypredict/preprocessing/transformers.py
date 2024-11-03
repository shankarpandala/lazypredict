from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    CustomTransformer for creating custom transformations that fit within scikit-learn pipelines.

    This class provides a template for creating custom transformers that apply specific transformations
    to the input data.

    Attributes
    ----------
    func : callable
        The custom function to apply to the data.

    Methods
    -------
    fit(X, y=None):
        Fits the custom transformer to the data (if necessary).
    transform(X):
        Transforms the data by applying the custom function.
    """

    def __init__(self, func):
        """
        Parameters
        ----------
        func : callable
            A custom function that applies a transformation to the data.
        """
        self.func = func

    def fit(self, X, y=None):
        """Fit the custom transformer to the data (if necessary)."""
        return self

    def transform(self, X):
        """Transform the data by applying the custom function."""
        return self.func(X)
