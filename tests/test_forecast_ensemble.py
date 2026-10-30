"""Forecast ensembles learn from the training series only (issue #601)."""

import numpy as np
import pytest

from lazypredict.TimeSeriesForecasting import LazyForecaster

pytest.importorskip("statsmodels")  # Holt / SARIMAX forecasters

MODELS = ["Naive", "Holt", "Ridge_TS", "SeasonalNaive"]


@pytest.fixture
def series():
    t = np.arange(160)
    y = 20 + 0.1 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(0).normal(0, 0.3, 160)
    return y[:140], y[140:]


def _fit(y_train, y_test):
    forecaster = LazyForecaster(forecasters=MODELS, seasonal_period=12)
    forecaster.fit(y_train, y_test)
    return forecaster


@pytest.mark.parametrize("method", ["weighted_average", "stacking"])
def test_ensemble_does_not_depend_on_test_values(series, method):
    y_train, y_test = series
    first = _fit(y_train, y_test).ensemble(method)
    second = _fit(y_train, y_test[::-1] * 3.0).ensemble(method)
    np.testing.assert_allclose(first, second)


def test_weighted_average_differs_from_simple_average(series):
    forecaster = _fit(*series)
    weighted = forecaster.ensemble("weighted_average")
    simple = forecaster.ensemble("simple_average")
    assert not np.allclose(weighted, simple)
    y_test = series[1]
    rmse = lambda pred: np.sqrt(np.mean((pred - y_test) ** 2))  # noqa: E731
    assert rmse(weighted) < rmse(simple)


def test_y_true_is_deprecated(series):
    forecaster = _fit(*series)
    with pytest.warns(FutureWarning, match="y_true"):
        forecaster.ensemble("stacking", y_true=series[1])


def test_too_short_training_series_raises():
    y = np.sin(np.arange(40) / 2.0)
    forecaster = LazyForecaster(forecasters=["Naive", "Holt"], n_lags=10, n_rolling=(3, 7))
    forecaster.fit(y[:19], y[19:])
    with pytest.raises(ValueError, match="too short"):
        forecaster.ensemble("stacking")


@pytest.mark.parametrize("method", ["weighted_average", "stacking"])
def test_one_step_test_horizon(series, method):
    # Issue #618: the validation window had the test length (1) and always failed.
    y_train, y_test = series
    forecaster = LazyForecaster(forecasters=MODELS, seasonal_period=12)
    forecaster.fit(y_train, y_test[:1])
    prediction = forecaster.ensemble(method)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
