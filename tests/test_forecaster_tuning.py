"""Forecaster tuning applies its parameters and is reproducible (issue #594)."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from lazypredict.ts_search_spaces import TS_SEARCH_SPACES
from lazypredict.ts_tuning import _apply_params_to_wrapper
from lazypredict.TimeSeriesForecasting import (
    HoltForecaster,
    HoltWintersForecaster,
    LazyForecaster,
    MLForecaster,
    SARIMAXForecaster,
    ThetaForecaster,
)

pytest.importorskip("optuna")
pytest.importorskip("statsmodels")


@pytest.fixture
def y():
    t = np.arange(160)
    return 50 + np.cumsum(np.random.default_rng(0).normal(size=160)) + 2 * np.sin(2 * np.pi * t / 12)


def test_ml_params_and_lag_features_are_applied(y):
    wrapper = MLForecaster(Ridge, "Ridge_TS")
    _apply_params_to_wrapper(wrapper, "Ridge_TS", {"alpha": 1e6, "n_lags": 5, "n_rolling_1": 2, "n_rolling_2": 6})
    wrapper.fit(y)
    assert wrapper._estimator.alpha == 1e6
    assert (wrapper.n_lags, wrapper.n_rolling) == (5, (2, 6))


def test_statistical_params_are_applied(y):
    sarimax = SARIMAXForecaster()
    _apply_params_to_wrapper(sarimax, "SARIMAX", {"order": (3, 0, 0), "seasonal_period": 12})
    sarimax.fit(y)
    assert sarimax._model.model.order == (3, 0, 0)
    assert sarimax.seasonal_period == 12

    holt = HoltForecaster()
    _apply_params_to_wrapper(holt, "Holt", {"smoothing_level": 0.2, "smoothing_trend": 0.05, "damped_trend": True})
    holt.fit(y)
    assert holt._model.params["smoothing_level"] == pytest.approx(0.2)
    assert holt._model.model.damped_trend

    winters = HoltWintersForecaster(seasonal_periods=4)
    _apply_params_to_wrapper(winters, "HoltWinters_Add", {"seasonal_period": 12, "smoothing_seasonal": 0.3})
    winters.fit(y)
    assert winters.seasonal_periods == 12
    assert winters._model.params["smoothing_seasonal"] == pytest.approx(0.3)


def test_theta_parameter_changes_the_forecast(y):
    low, high = ThetaForecaster(), ThetaForecaster()
    _apply_params_to_wrapper(low, "Theta", {"theta": 1.0})
    _apply_params_to_wrapper(high, "Theta", {"theta": 4.0})
    low.fit(y)
    high.fit(y)
    assert not np.allclose(low.predict(5), high.predict(5))


def test_theta_search_space_respects_statsmodels_bounds():
    class Trial:
        def suggest_float(self, name, low, high, **kwargs):
            return low

    assert TS_SEARCH_SPACES["Theta"](Trial())["theta"] >= 1.0


def test_tuning_is_reproducible_and_used(y):
    def run():
        forecaster = LazyForecaster(forecasters=["Ridge_TS", "Naive"], tune=True, tune_top_k=2,
                                    tune_trials=4, random_state=3)
        forecaster.fit(y[:140], y[140:])
        return forecaster

    first, second = run(), run()
    assert first.tuned_scores_.loc["Ridge_TS", "Best Params"] == second.tuned_scores_.loc["Ridge_TS", "Best Params"]
    table = first.tuned_scores_
    assert list(table.columns) == ["Best RMSE", "Best Params", "Status"]
    assert table.loc["Naive", "Status"] == "no search space"
    params = table.loc["Ridge_TS", "Best Params"]
    assert first.tuned_models_["Ridge_TS"]._estimator.alpha == params["alpha"]
