"""LazyForecaster model selection, fit() state and timeouts (issues #602, #590)."""

import signal
import time

import numpy as np
import pytest

import lazypredict.TimeSeriesForecasting as tsf
from lazypredict.exceptions import TimeoutException
from lazypredict.TimeSeriesForecasting import LazyForecaster


@pytest.fixture
def series():
    t = np.arange(140)
    y = 10 + 0.05 * t + np.sin(t / 3.0) + np.random.default_rng(0).normal(0, 0.2, 140)
    return y[:120], y[120:]


def test_single_name_selects_that_forecaster(series):
    scores, _ = LazyForecaster(forecasters="Holt").fit(*series)
    assert list(scores.index) == ["Holt"]


@pytest.mark.parametrize("selection", [["Holt", "Prophet"], ["RandomForest_TS"]])
def test_unknown_names_raise(series, selection):
    with pytest.raises(ValueError, match="Unknown forecaster name"):
        LazyForecaster(forecasters=selection).fit(*series)


def test_unknown_name_suggests_the_real_one(series):
    with pytest.raises(ValueError, match="did you mean 'RandomForestRegressor_TS'"):
        LazyForecaster(forecasters=["RandomForestRegressor"]).fit(*series)


def test_refit_does_not_keep_previous_models(series):
    forecaster = LazyForecaster(forecasters=["Naive", "Holt", "Ridge_TS"])
    forecaster.fit(*series)
    forecaster.forecasters = ["Naive"]
    forecaster.fit(*series)
    assert list(forecaster.models) == ["Naive"]
    assert list(forecaster.predict(series[0], 3)) == ["Naive"]


def test_models_that_fail_to_predict_are_not_registered(series, monkeypatch):
    def broken_predict(self, horizon, X_test=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(tsf.HoltForecaster, "predict", broken_predict)
    forecaster = LazyForecaster(forecasters=["Naive", "Holt"])
    scores, _ = forecaster.fit(*series)
    assert list(scores.index) == ["Naive"]
    assert "Holt" in forecaster.errors and "Holt" not in forecaster.models


def test_ensemble_before_fit_raises_value_error():
    with pytest.raises(ValueError, match="Call fit"):
        LazyForecaster().ensemble("simple_average")


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer (POSIX)")
def test_slow_forecaster_is_interrupted(series, monkeypatch):
    def slow_fit(self, y_train, X_train=None):
        time.sleep(5)

    monkeypatch.setattr(tsf.NaiveForecaster, "fit", slow_fit)
    forecaster = LazyForecaster(forecasters=["Naive", "Holt"], timeout=0.5)
    start = time.perf_counter()
    scores, _ = forecaster.fit(*series)
    assert time.perf_counter() - start < 3
    assert list(scores.index) == ["Holt"]
    assert isinstance(forecaster.errors["Naive"], TimeoutException)
