"""Direct and multi-output horizon strategies (issue #595)."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from lazypredict.horizon import direct_forecast, multi_output_forecast
from lazypredict.TimeSeriesForecasting import LazyForecaster


@pytest.mark.parametrize("strategy", [direct_forecast, multi_output_forecast])
def test_strategies_forecast_the_next_steps(strategy):
    y = np.arange(100.0)
    pred = strategy(LinearRegression, y, horizon=5, n_lags=5, n_rolling=(3,))
    np.testing.assert_allclose(pred, [100, 101, 102, 103, 104], atol=1e-6)


@pytest.mark.parametrize("strategy", [direct_forecast, multi_output_forecast])
def test_too_long_horizons_raise_instead_of_zero_filling(strategy):
    with pytest.raises(ValueError, match="Not enough history"):
        strategy(LinearRegression, np.arange(40.0), horizon=40, n_lags=5, n_rolling=(3,))


@pytest.mark.parametrize("strategy", [direct_forecast, multi_output_forecast])
def test_exogenous_variables(strategy):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(130, 1))
    y = 10 + 3 * X[:, 0] + rng.normal(scale=0.01, size=130)
    pred = strategy(LinearRegression, y[:120], horizon=1, n_lags=3, n_rolling=(3,),
                    X_exog=X[:120], X_future=X[120:])
    assert abs(pred[0] - y[120]) < 0.1
    with pytest.raises(ValueError, match="X_future"):
        strategy(LinearRegression, y[:120], horizon=1, n_lags=3, n_rolling=(3,), X_exog=X[:120])


def test_lazy_forecaster_uses_the_requested_strategy():
    t = np.arange(150)
    y = 20 + np.sin(t / 2.0) * 5 + np.random.default_rng(1).normal(0, 0.5, 150)
    preds = {}
    for strategy in ("recursive", "direct", "multi_output"):
        forecaster = LazyForecaster(forecasters=["Ridge_TS"], horizon_strategy=strategy, predictions=True)
        _, predictions = forecaster.fit(y[:130], y[130:])
        preds[strategy] = predictions["Ridge_TS"].to_numpy()
    assert not np.allclose(preds["recursive"], preds["direct"])
    assert not np.allclose(preds["recursive"], preds["multi_output"])


def test_n_jobs_is_validated_and_used_for_cv():
    with pytest.raises(ValueError, match="n_jobs"):
        LazyForecaster(n_jobs="garbage")
    y = 20 + np.sin(np.arange(120) / 2.0) * 5
    serial = LazyForecaster(forecasters=["Ridge_TS", "Naive"], cv=3, n_jobs=1).fit(y[:100], y[100:])[0]
    parallel = LazyForecaster(forecasters=["Ridge_TS", "Naive"], cv=3, n_jobs=2).fit(y[:100], y[100:])[0]
    cv_columns = [c for c in serial.columns if "CV" in c]
    np.testing.assert_allclose(serial[cv_columns].to_numpy(), parallel.loc[serial.index, cv_columns].to_numpy())
