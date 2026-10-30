"""AutoARIMA uses exogenous variables and the seasonal period (issue #596)."""

import numpy as np
import pytest

pytest.importorskip("pmdarima")

from lazypredict.TimeSeriesForecasting import AutoARIMAForecaster  # noqa: E402


def test_exogenous_variables_are_used():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(140, 1))
    y = 50 + 5 * X[:, 0] + rng.normal(scale=0.1, size=140)
    model = AutoARIMAForecaster()
    model.fit(y[:120], X[:120])
    pred = model.predict(20, X[120:])
    rmse = np.sqrt(np.mean((pred - y[120:]) ** 2))
    assert rmse < 0.5, rmse  # 5.26 when X was ignored


def test_seasonal_period_is_passed_as_m():
    t = np.arange(96)
    y = 10 + 3 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(0, 0.3, 96)
    model = AutoARIMAForecaster(seasonal_period=12)
    model.fit(y)
    assert model._model.seasonal_order[-1] == 12


def test_long_periods_fall_back_to_non_seasonal():
    model = AutoARIMAForecaster(seasonal_period=365)
    model.fit(np.random.default_rng(2).normal(size=60).cumsum())
    assert model._model.seasonal_order[-1] in (0, 1)


def test_models_saved_by_0_3_still_fit():
    # Issue #628: 0.3 objects have neither ``seasonal_period`` nor ``params``.
    model = AutoARIMAForecaster(seasonal_period=4)
    del model.seasonal_period, model.params
    y = 10 + np.sin(np.arange(60) / 2.0) + np.random.default_rng(0).normal(0, 0.1, 60)
    model.fit(y)
    assert np.isfinite(model.predict(5)).all()
