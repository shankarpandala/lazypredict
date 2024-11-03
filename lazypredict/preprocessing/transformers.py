# lazypredict/preprocessing/transformers.py

from ..utils.backend import Backend

# Usage of Backend to dynamically create DataFrames and Series
class CustomTransformer:
    """
    A custom transformer for demonstration.
    """

    def __init__(self):
        pass

    def transform(self, X):
        # Create DataFrame using Backend
        X_df = Backend.DataFrame(X)
        # Perform any transformation logic
        return X_df
