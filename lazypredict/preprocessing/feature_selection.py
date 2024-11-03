from sklearn.feature_selection import SelectKBest, f_classif
from .base import PreprocessingBase

class FeatureSelection(PreprocessingBase):
    """
    FeatureSelection for selecting the most relevant features for the model.

    This class uses scikit-learn's SelectKBest method by default to select the top features
    based on a scoring function.

    Attributes
    ----------
    k : int
        Number of top features to select.
    method : callable
        The scoring function used for feature selection, e.g., f_classif.

    Methods
    -------
    fit(X, y):
        Fits the feature selection component to the data.
    transform(X):
        Transforms the data by selecting the most relevant features.
    """

    def __init__(self, k=10, method=f_classif):
        """
        Parameters
        ----------
        k : int, optional
            Number of top features to select. Default is 10.
        method : callable, optional
            Scoring function used for feature selection. Default is f_classif.
        """
        self.k = k
        self.method = method
        self.selector = None

    def fit(self, X, y):
        """Fit the feature selection method to the data."""
        self.selector = SelectKBest(self.method, k=self.k)
        self.selector.fit(X, y)

    def transform(self, X):
        """Transform the data by selecting the most relevant features."""
        return X.iloc[:, self.selector.get_support(indices=True)]
