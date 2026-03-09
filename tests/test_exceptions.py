"""Tests for the custom exception hierarchy."""

import pytest
from lazypredict.exceptions import (
    DataValidationError,
    InvalidParameterError,
    LazyPredictError,
    ModelFitError,
    TimeoutException,
)


def test_base_exception():
    with pytest.raises(LazyPredictError):
        raise LazyPredictError("test error")


def test_timeout_exception_is_lazy_predict_error():
    exc = TimeoutException("timed out")
    assert isinstance(exc, LazyPredictError)


def test_model_fit_error():
    original = ValueError("bad value")
    exc = ModelFitError("LinearRegression", original)
    assert exc.model_name == "LinearRegression"
    assert exc.original_error is original
    assert "LinearRegression" in str(exc)
    assert isinstance(exc, LazyPredictError)


def test_invalid_parameter_error_is_value_error():
    exc = InvalidParameterError("bad param")
    assert isinstance(exc, ValueError)
    assert isinstance(exc, LazyPredictError)


def test_data_validation_error():
    exc = DataValidationError("empty data")
    assert isinstance(exc, ValueError)
    assert isinstance(exc, LazyPredictError)
