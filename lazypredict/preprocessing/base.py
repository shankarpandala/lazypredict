from abc import ABC, abstractmethod

class PreprocessingBase(ABC):
    """
    Abstract base class for preprocessing components in LazyPredict.

    This class provides a common interface for `fit` and `transform` methods.
    Concrete classes should implement these methods to apply specific preprocessing steps.

    Methods
    -------
    fit(X, y=None):
        Fits the preprocessing component to the data.
    transform(X):
        Transforms the data based on the fitted component.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the preprocessing component to the data."""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform the data based on the fitted component."""
        pass
