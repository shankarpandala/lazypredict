"""Tests for LazyForecaster time series forecasting module."""

import numpy as np
import pandas as pd
import pytest

from lazypredict.TimeSeriesForecasting import (
    LazyForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MLForecaster,
    ForecasterWrapper,
)
from lazypredict.metrics import (
    compute_forecast_metrics,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    symmetric_mean_absolute_percentage_error,
)
from lazypredict.ts_preprocessing import (
    create_lag_features,
    detect_seasonal_period,
    recursive_forecast,
)
from lazypredict.exceptions import InsufficientDataError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def univariate_data():
    """Simple trend + noise data."""
    np.random.seed(42)
    n = 200
    y = np.cumsum(np.random.randn(n)) + 100
    return y[:180], y[180:]


@pytest.fixture
def seasonal_data():
    """Synthetic data with clear seasonality (period=12)."""
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=float)
    y = 50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n)
    return y[:180], y[180:]


@pytest.fixture
def exogenous_data():
    """Univariate data with exogenous features."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y = 50 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n) * 0.5
    return y[:180], y[180:], X[:180], X[180:]


# ---------------------------------------------------------------------------
# Metric Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mape_basic(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert 0 < result < 100

    def test_mape_with_zeros(self):
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 190.0])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert np.isfinite(result)  # zeros excluded, not inf

    def test_mape_all_zeros(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert result == np.inf

    def test_smape_basic(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        result = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        assert 0 < result < 200

    def test_smape_symmetric(self):
        y_true = np.array([100.0])
        y_pred = np.array([200.0])
        result1 = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        result2 = symmetric_mean_absolute_percentage_error(y_pred, y_true)
        assert abs(result1 - result2) < 1e-10

    def test_mase_basic(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        y_pred = np.array([5.5, 7.5])
        result = mean_absolute_scaled_error(y_true, y_pred, y_train)
        assert result > 0

    def test_compute_forecast_metrics_keys(self):
        y_train = np.arange(50, dtype=float)
        y_true = np.array([50.0, 51.0, 52.0])
        y_pred = np.array([49.0, 52.0, 53.0])
        result = compute_forecast_metrics(y_true, y_pred, y_train)
        expected_keys = {"mae", "rmse", "r_squared", "mape", "smape", "mase"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_create_lag_features_shape(self):
        y = np.arange(50, dtype=float)
        X, y_target = create_lag_features(y, n_lags=5, n_rolling=(3,))
        assert X.shape[0] == y_target.shape[0]
        assert X.shape[0] > 0
        # 5 lags + 1 rolling mean + 1 rolling std + 1 diff = 8 features
        assert X.shape[1] == 8

    def test_create_lag_features_with_exog(self):
        y = np.arange(50, dtype=float)
        X_exog = np.random.randn(50, 2)
        X, y_target = create_lag_features(y, n_lags=5, n_rolling=(3,), X_exog=X_exog)
        # 8 base features + 2 exogenous
        assert X.shape[1] == 10

    def test_detect_seasonal_period_with_seasonality(self):
        np.random.seed(42)
        t = np.arange(200, dtype=float)
        y = 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 0.5
        period = detect_seasonal_period(y)
        # Should detect period near 12
        assert period is not None
        assert 10 <= period <= 14

    def test_detect_seasonal_period_no_seasonality(self):
        np.random.seed(42)
        y = np.random.randn(100)
        period = detect_seasonal_period(y)
        # May or may not detect, but shouldn't crash
        assert period is None or isinstance(period, int)

    def test_detect_seasonal_period_short_series(self):
        y = np.array([1.0, 2.0, 3.0])
        period = detect_seasonal_period(y)
        assert period is None


# ---------------------------------------------------------------------------
# Wrapper Tests
# ---------------------------------------------------------------------------

class TestWrappers:
    def test_naive_forecaster(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = NaiveForecaster()
        f.fit(y_train)
        pred = f.predict(3)
        np.testing.assert_array_equal(pred, [5.0, 5.0, 5.0])

    def test_seasonal_naive_forecaster(self):
        y_train = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
        f = SeasonalNaiveForecaster(seasonal_period=3)
        f.fit(y_train)
        pred = f.predict(6)
        expected = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(pred, expected)

    def test_ml_forecaster(self, univariate_data):
        from sklearn.linear_model import Ridge
        y_train, y_test = univariate_data
        f = MLForecaster(
            estimator_class=Ridge,
            model_name="Ridge_TS",
            n_lags=5,
            n_rolling=(3,),
        )
        f.fit(y_train)
        pred = f.predict(len(y_test))
        assert len(pred) == len(y_test)
        assert all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# LazyForecaster Tests
# ---------------------------------------------------------------------------

class TestLazyForecasterBasic:
    def test_basic_fit(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            verbose=0,
            ignore_warnings=True,
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, predictions = forecaster.fit(y_train, y_test)
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) >= 1
        assert "MAE" in scores.columns
        assert "RMSE" in scores.columns
        assert "MAPE" in scores.columns
        assert "SMAPE" in scores.columns
        assert "MASE" in scores.columns
        assert "R-Squared" in scores.columns
        assert "Time Taken" in scores.columns

    def test_fit_returns_two_dataframes(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive"],
        )
        result = forecaster.fit(y_train, y_test)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predictions_flag(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            predictions=True,
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, preds = forecaster.fit(y_train, y_test)
        assert not preds.empty
        assert preds.shape[0] == len(y_test)

    def test_predictions_flag_false(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            predictions=False,
            forecasters=["Naive"],
        )
        scores, preds = forecaster.fit(y_train, y_test)
        assert preds.empty

    def test_specific_forecasters(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive", "SeasonalNaive"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) == 2
        assert "Naive" in scores.index
        assert "SeasonalNaive" in scores.index

    def test_max_models(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(max_models=3)
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) <= 3

    def test_sort_by_rmse(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            sort_by="RMSE",
            forecasters=["Naive", "Ridge_TS", "Lasso_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        if len(scores) > 1:
            rmse_values = scores["RMSE"].values
            assert all(rmse_values[i] <= rmse_values[i + 1] for i in range(len(rmse_values) - 1))

    def test_sort_by_r_squared(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            sort_by="R-Squared",
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        if len(scores) > 1:
            r2_values = scores["R-Squared"].values
            # R-Squared sorted descending (higher is better)
            assert all(r2_values[i] >= r2_values[i + 1] for i in range(len(r2_values) - 1))


class TestLazyForecasterExogenous:
    def test_with_exogenous(self, exogenous_data):
        y_train, y_test, X_train, X_test = exogenous_data
        forecaster = LazyForecaster(
            forecasters=["Ridge_TS", "Naive"],
        )
        scores, _ = forecaster.fit(y_train, y_test, X_train, X_test)
        assert len(scores) >= 1


class TestLazyForecasterCV:
    def test_cv_columns_present(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            cv=3,
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert "MAE CV Mean" in scores.columns
        assert "RMSE CV Mean" in scores.columns
        assert "MAE CV Std" in scores.columns


class TestLazyForecasterCustomMetric:
    def test_custom_metric_appears(self, univariate_data):
        y_train, y_test = univariate_data

        def my_metric(y_true, y_pred):
            return float(np.mean(np.abs(y_true - y_pred)))

        forecaster = LazyForecaster(
            custom_metric=my_metric,
            forecasters=["Naive"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert "my_metric" in scores.columns


class TestLazyForecasterSeasonal:
    def test_seasonal_period_override(self, seasonal_data):
        y_train, y_test = seasonal_data
        forecaster = LazyForecaster(
            seasonal_period=12,
            forecasters=["SeasonalNaive", "Naive"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1

    def test_seasonal_auto_detect(self, seasonal_data):
        y_train, y_test = seasonal_data
        forecaster = LazyForecaster(
            seasonal_period=None,  # auto-detect
            forecasters=["Naive"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1


class TestLazyForecasterTimeout:
    def test_timeout_skips_slow_models(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            timeout=0.0001,  # extremely short — most models will be skipped
            forecasters=["Naive", "Ridge_TS", "RandomForestRegressor_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        # Some models may be skipped; at minimum Naive should survive
        assert isinstance(scores, pd.DataFrame)


class TestLazyForecasterErrorHandling:
    def test_errors_stored(self, univariate_data):
        y_train, y_test = univariate_data
        # Use a very short series to trigger errors in some models
        forecaster = LazyForecaster(
            ignore_warnings=True,
            forecasters=["Naive"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        # errors dict should be accessible
        assert isinstance(forecaster.errors, dict)

    def test_models_stored(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(forecaster.models) >= 1


class TestLazyForecasterInputValidation:
    def test_empty_y_train(self):
        forecaster = LazyForecaster()
        with pytest.raises(ValueError, match="y_train is empty"):
            forecaster.fit(np.array([]), np.array([1.0, 2.0]))

    def test_empty_y_test(self):
        forecaster = LazyForecaster()
        with pytest.raises(ValueError, match="y_test is empty"):
            forecaster.fit(np.array([1.0, 2.0, 3.0]), np.array([]))

    def test_insufficient_data(self):
        forecaster = LazyForecaster(n_lags=50)
        with pytest.raises(InsufficientDataError):
            forecaster.fit(np.arange(20, dtype=float), np.array([1.0, 2.0]))


class TestLazyForecasterInitValidation:
    def test_invalid_cv(self):
        with pytest.raises(ValueError):
            LazyForecaster(cv=1)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError):
            LazyForecaster(timeout=-1)

    def test_invalid_custom_metric(self):
        with pytest.raises(TypeError):
            LazyForecaster(custom_metric="not_callable")

    def test_invalid_max_models(self):
        with pytest.raises(ValueError):
            LazyForecaster(max_models=0)


class TestLazyForecasterProvideModels:
    def test_provide_models(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive", "Ridge_TS"],
        )
        models = forecaster.provide_models(y_train, y_test)
        assert isinstance(models, dict)
        assert len(models) >= 1
        for name, wrapper in models.items():
            assert isinstance(wrapper, ForecasterWrapper)


class TestLazyForecasterWithPandasInput:
    def test_pandas_series_input(self, univariate_data):
        y_train, y_test = univariate_data
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        forecaster = LazyForecaster(
            forecasters=["Naive"],
        )
        scores, _ = forecaster.fit(y_train_series, y_test_series)
        assert len(scores) >= 1


class TestLazyForecasterProgressCallback:
    def test_callback_called(self, univariate_data):
        y_train, y_test = univariate_data
        callback_log = []

        def my_callback(name, current, total, metrics):
            callback_log.append((name, current, total))

        forecaster = LazyForecaster(
            progress_callback=my_callback,
            forecasters=["Naive", "Ridge_TS"],
        )
        forecaster.fit(y_train, y_test)
        assert len(callback_log) >= 1


# ---------------------------------------------------------------------------
# Optional dependency tests (statsmodels)
# ---------------------------------------------------------------------------

try:
    import statsmodels  # noqa: F401
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@pytest.mark.skipif(not _HAS_STATSMODELS, reason="statsmodels not installed")
class TestStatsmodelsForecasters:
    def test_simple_exp_smoothing(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["SimpleExpSmoothing"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1

    def test_holt(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Holt"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1

    def test_theta(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Theta"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1

    def test_holt_winters_with_seasonal_period(self, seasonal_data):
        y_train, y_test = seasonal_data
        forecaster = LazyForecaster(
            seasonal_period=12,
            forecasters=["HoltWinters_Add"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1

    def test_sarimax(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["SARIMAX"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1


# ---------------------------------------------------------------------------
# Optional dependency tests (pmdarima)
# ---------------------------------------------------------------------------

try:
    import pmdarima  # noqa: F401
    _HAS_PMDARIMA = True
except ImportError:
    _HAS_PMDARIMA = False


@pytest.mark.skipif(not _HAS_PMDARIMA, reason="pmdarima not installed")
class TestPmdarimaForecasters:
    def test_auto_arima(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["AutoARIMA"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1


# ---------------------------------------------------------------------------
# Optional dependency tests (torch)
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
class TestDeepLearningForecasters:
    def test_lstm(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["LSTM_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1

    def test_gru(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["GRU_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1


# ---------------------------------------------------------------------------
# Optional dependency tests (timesfm)
# ---------------------------------------------------------------------------

try:
    import timesfm  # noqa: F401
    _HAS_TIMESFM = True
except ImportError:
    _HAS_TIMESFM = False


@pytest.mark.skipif(not _HAS_TIMESFM, reason="timesfm not installed")
class TestTimesFMForecaster:
    def test_timesfm_basic(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["TimesFM"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        assert len(scores) >= 1
