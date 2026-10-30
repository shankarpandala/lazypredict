"""Adjusted R², MASE and exception pickling edge cases (issue #606)."""

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_regression

from lazypredict.exceptions import ModelFitError
from lazypredict.metrics import adjusted_rsquared, mean_absolute_scaled_error
from lazypredict.Supervised import LazyRegressor

REGRESSORS = ["LinearRegression", "Ridge", "DummyRegressor"]


@pytest.mark.parametrize("n, p", [(10, 9), (10, 20)])
def test_adjusted_rsquared_undefined(n, p):
    assert np.isnan(adjusted_rsquared(0.5, n=n, p=p))


def test_adjusted_rsquared_regular_case():
    assert adjusted_rsquared(0.5, n=101, p=10) == pytest.approx(1 - 0.5 * 100 / 90)


def test_small_test_set_does_not_fail_every_model():
    X, y = make_regression(200, 9, noise=5, random_state=0)
    reg = LazyRegressor(regressors=REGRESSORS)
    scores, _ = reg.fit(X[:190], X[190:], y[:190], y[190:])  # 10 rows, 9 features
    assert not reg.errors
    assert scores["Adjusted R-Squared"].isna().all()
    assert scores.index[-1] == "DummyRegressor"  # ranked by R² instead


def test_more_features_than_test_rows_keeps_sensible_ranking():
    X, y = make_regression(100, 20, noise=5, random_state=0)
    scores, _ = LazyRegressor(regressors=REGRESSORS).fit(X[:90], X[90:], y[:90], y[90:])
    assert scores.index[-1] == "DummyRegressor"
    assert (scores["R-Squared"].diff().dropna() <= 0).all()


def test_mase_short_training_series_falls_back_to_lag_one():
    y_train = np.array([1.0, 2.0, 4.0])
    value = mean_absolute_scaled_error([5.0], [4.0], y_train, seasonal_period=12)
    assert value == pytest.approx(1.0 / 1.5)


def test_mase_rejects_invalid_period():
    with pytest.raises(ValueError, match="seasonal_period"):
        mean_absolute_scaled_error([1.0], [1.0], [1.0, 2.0], seasonal_period=0)


def test_model_fit_error_round_trips_through_pickle():
    error = ModelFitError("SVC", ValueError("bad input"))
    restored = pickle.loads(pickle.dumps(error))
    assert restored.model_name == "SVC"
    assert isinstance(restored.original_error, ValueError)
    assert str(restored) == "Model 'SVC' failed: bad input"
