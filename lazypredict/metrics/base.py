from abc import ABC, abstractmethod

class MetricsBase(ABC):
    """
    Abstract base class for metrics in LazyPredict.

    This class provides a common interface for computing and displaying metrics.
    Concrete classes should implement the `compute` method.

    Methods
    -------
    compute(y_true, y_pred):
        Compute metrics based on true and predicted values.
    """

    @abstractmethod
    def compute(self, y_true, y_pred):
        """Compute metrics based on true and predicted values."""
        pass
