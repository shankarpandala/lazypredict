import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from .base import PreprocessingBase

class FeatureEngineering(PreprocessingBase):
    """
    FeatureEngineering for creating new features through various transformations.

    This class allows for applying polynomial transformations, interaction terms, and other custom
    feature engineering techniques to enhance model performance.

    Attributes
    ----------
    method : str
        The type of feature engineering to apply, e.g., 'polynomial'.

    Methods
    -------
    fit(X, y=None):
        Fits the feature engineering component to the data (if necessary).
    transform(X):
        Transforms the data by adding new features based on the specified method.
    """

    def __init__(self, method='polynomial', degree=2):
        """
        Parameters
        ----------
        method : str, optional
            Type of feature engineering to apply. Default is 'polynomial'.
        degree : int, optional
            Degree of polynomial features. Used only if `method` is 'polynomial'. Default is 2.
        """
        self.method = method
        self.degree = degree
        self.poly = None

    def fit(self, X, y=None):
        """Fit the feature engineering method to the data."""
        if self.method == 'polynomial':
            self.poly = PolynomialFeatures(self.degree)
            self.poly.fit(X)

    def transform(self, X):
        """Transform the data by adding engineered features."""
        if self.method == 'polynomial':
            return pd.DataFrame(self.poly.transform(X), columns=self.poly.get_feature_names_out())
        return X
