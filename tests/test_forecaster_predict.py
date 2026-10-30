"""LazyForecaster.predict() with exogenous variables (issue #599)."""

import numpy as np
import pytest

from lazypredict.TimeSeriesForecasting import LazyForecaster

pytest.importorskip("statsmodels")  # Holt / SARIMAX forecasters


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(140, 1))
    y = 50 + 5 * X[:, 0] + rng.normal(scale=0.1, size=140)
    forecaster = LazyForecaster(forecasters=["Ridge_TS", "SARIMAX", "Naive"])
    forecaster.fit(y[:120], y[120:], X[:120], X[120:])
    return forecaster, y, X


@pytest.mark.parametrize("name", ["Ridge_TS", "SARIMAX"])
def test_predict_uses_exogenous_history_and_future(fitted, name):
    forecaster, y, X = fitted
    pred = forecaster.predict(y[:120], 20, model_name=name, X_test=X[120:], X_history=X[:120])
    assert np.sqrt(np.mean((pred - y[120:]) ** 2)) < 0.5


def test_predict_requires_exogenous_inputs_when_fitted_with_them(fitted):
    forecaster, y, X = fitted
    with pytest.raises(ValueError, match="fitted with exogenous variables"):
        forecaster.predict(y[:120], 20)
    with pytest.raises(ValueError, match="passed together"):
        forecaster.predict(y[:120], 20, X_test=X[120:])
    with pytest.raises(ValueError, match="X_test has 20 rows but the forecast horizon has 10"):
        forecaster.predict(y[:120], 10, X_test=X[120:], X_history=X[:120])


def test_predict_does_not_modify_the_fitted_models(fitted):
    forecaster, y, X = fitted
    stored = forecaster.models["Naive"]
    before = stored._last_value
    forecaster.predict(y[:100] + 1000, 5, X_test=X[100:105], X_history=X[:100])
    assert forecaster.models["Naive"] is stored
    assert stored._last_value == before


def test_predict_without_exogenous_variables():
    y = np.sin(np.arange(130) / 3.0) + np.arange(130) * 0.02
    forecaster = LazyForecaster(forecasters=["Naive", "Holt"])
    forecaster.fit(y[:110], y[110:])
    preds = forecaster.predict(y, 7)
    assert set(preds) == {"Naive", "Holt"} and all(len(p) == 7 for p in preds.values())
    with pytest.raises(ValueError, match="horizon"):
        forecaster.predict(y, 0)
