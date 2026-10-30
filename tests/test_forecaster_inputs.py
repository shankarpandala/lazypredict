"""LazyForecaster input validation (issues #598, #603)."""

import functools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_pinball_loss

from lazypredict.ts_preprocessing import create_lag_features
from lazypredict.TimeSeriesForecasting import LazyForecaster

pytest.importorskip("statsmodels")  # Holt / SARIMAX forecasters


@pytest.fixture
def exog_series():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(140, 1))
    y = 50 + 5 * X[:, 0] + rng.normal(scale=0.1, size=140)
    return y, X


# -- #598 ---------------------------------------------------------------------

def test_exogenous_length_mismatch_raises(exog_series):
    y, X = exog_series
    with pytest.raises(ValueError, match="X_train has 150 rows but y_train has 120"):
        LazyForecaster(forecasters=["Ridge_TS"]).fit(y[:120], y[120:], np.vstack([X, X])[:150], X[120:])


def test_exogenous_variables_must_be_passed_together(exog_series):
    y, X = exog_series
    with pytest.raises(ValueError, match="passed together"):
        LazyForecaster(forecasters=["Ridge_TS"]).fit(y[:120], y[120:], X[:120], None)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_targets_raise(exog_series, bad):
    y, _ = exog_series
    y = y.copy()
    y[60] = bad
    with pytest.raises(ValueError, match="y_train contains NaN or infinite"):
        LazyForecaster(forecasters=["Naive"]).fit(y[:120], y[120:])


def test_aligned_exogenous_features_are_used(exog_series):
    y, X = exog_series
    scores, _ = LazyForecaster(forecasters=["Ridge_TS"]).fit(y[:120], y[120:], X[:120], X[120:])
    assert scores.loc["Ridge_TS", "RMSE"] < 0.5


def test_lag_features_keep_exogenous_rows_aligned():
    y = np.arange(30.0)
    X = np.arange(30.0).reshape(-1, 1) * 10  # exog value = 10 * time index
    features, target = create_lag_features(y, n_lags=3, n_rolling=(3,), X_exog=X)
    np.testing.assert_allclose(features[:, -1], target * 10)


# -- #603 ---------------------------------------------------------------------

def test_partial_custom_metric(exog_series):
    y, _ = exog_series
    metric = functools.partial(mean_pinball_loss, alpha=0.9)
    scores, _ = LazyForecaster(forecasters=["Naive"], custom_metric=metric).fit(y[:120], y[120:])
    assert "mean_pinball_loss" in scores.columns


def test_one_column_dataframe_target(exog_series):
    y, _ = exog_series
    frame = pd.DataFrame({"sales": y})
    scores, _ = LazyForecaster(forecasters=["Naive", "Holt"]).fit(frame.iloc[:120], frame.iloc[120:])
    assert len(scores) == 2


def test_wide_dataframe_target_raises(exog_series):
    y, _ = exog_series
    frame = pd.DataFrame({"a": y, "b": y})
    with pytest.raises(ValueError, match="single series"):
        LazyForecaster(forecasters=["Naive"]).fit(frame.iloc[:120], frame.iloc[120:])


def test_sort_by_is_case_insensitive_and_validated(exog_series):
    y, _ = exog_series
    forecaster = LazyForecaster(forecasters=["Naive", "Holt"], sort_by="rmse")
    assert forecaster.sort_by == "RMSE"
    scores, _ = forecaster.fit(y[:120], y[120:])
    assert scores["RMSE"].is_monotonic_increasing
    with pytest.raises(ValueError, match="sort_by must be one of"):
        LazyForecaster(sort_by="accuracy")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tune_top_k": 0}, "tune_top_k"),
        ({"tune_trials": 0}, "tune_trials"),
        ({"seasonal_period": 0}, "seasonal_period"),
    ],
)
def test_options_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LazyForecaster(**kwargs)
