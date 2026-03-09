"""Custom exception types for LazyPredict."""


class LazyPredictError(Exception):
    """Base exception for all LazyPredict errors."""


class TimeoutException(LazyPredictError):
    """Raised when a model exceeds the allotted time limit."""


class ModelFitError(LazyPredictError):
    """Raised when a model fails during fitting."""

    def __init__(self, model_name: str, original_error: Exception):
        self.model_name = model_name
        self.original_error = original_error
        super().__init__(f"Model '{model_name}' failed: {original_error}")


class InvalidParameterError(LazyPredictError, ValueError):
    """Raised when an invalid parameter is passed to a constructor or method."""


class DataValidationError(LazyPredictError, ValueError):
    """Raised when input data fails validation checks."""
