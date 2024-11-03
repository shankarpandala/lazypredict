# lazypredict/preprocessing/feature_selection.py

from ..utils.backend import Backend

class FeatureSelector:
    """
    FeatureSelector class for selecting features.
    """

    def __init__(self, params=None):
        self.params = params or {}

    def fit(self, X, y=None):
        # Use Backend to create the appropriate DataFrame
        X_df = Backend.DataFrame(X)
        # Add your feature selection logic here
        return self
