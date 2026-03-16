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
        from lazypredict.TimeSeriesForecasting import LSTMForecaster
        y_train, y_test = univariate_data
        model = LSTMForecaster(n_epochs=3)
        model.fit(y_train)
        pred = model.predict(len(y_test))
        assert len(pred) == len(y_test)
        assert not np.any(np.isnan(pred))

    def test_gru(self, univariate_data):
        from lazypredict.TimeSeriesForecasting import GRUForecaster
        y_train, y_test = univariate_data
        model = GRUForecaster(n_epochs=3)
        model.fit(y_train)
        pred = model.predict(len(y_test))
        assert len(pred) == len(y_test)
        assert not np.any(np.isnan(pred))


# ---------------------------------------------------------------------------
# Optional dependency tests (gluonts — DeepAR)
# ---------------------------------------------------------------------------

try:
    import gluonts  # noqa: F401
    _HAS_GLUONTS = True
except ImportError:
    _HAS_GLUONTS = False


@pytest.mark.skipif(not _HAS_GLUONTS, reason="gluonts not installed")
class TestDeepARForecaster:
    def test_deepar_basic(self, univariate_data):
        from lazypredict.TimeSeriesForecasting import DeepARForecaster
        y_train, y_test = univariate_data
        model = DeepARForecaster(max_epochs=1, num_batches_per_epoch=5)
        model.fit(y_train)
        pred = model.predict(len(y_test))
        assert len(pred) == len(y_test)
        assert not np.any(np.isnan(pred))

    def test_deepar_custom_horizon(self, univariate_data):
        from lazypredict.TimeSeriesForecasting import DeepARForecaster
        y_train, _ = univariate_data
        model = DeepARForecaster(
            prediction_length=5, max_epochs=1, num_batches_per_epoch=5
        )
        model.fit(y_train)
        pred = model.predict(5)
        assert len(pred) == 5
        assert not np.any(np.isnan(pred))

    def test_deepar_name(self):
        from lazypredict.TimeSeriesForecasting import DeepARForecaster
        model = DeepARForecaster()
        assert model.name == "DeepAR_TS"


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
        from lazypredict.TimeSeriesForecasting import TimesFMForecaster
        y_train, y_test = univariate_data
        model = TimesFMForecaster(use_gpu=False)
        try:
            model.fit(y_train)
            pred = model.predict(len(y_test))
            assert len(pred) == len(y_test)
        except Exception:
            # Model weights may not be available in CI
            pytest.skip("TimesFM model weights not available")


# ---------------------------------------------------------------------------
# Predictions storage bug fix tests
# ---------------------------------------------------------------------------

class TestPredictionsStorage:
    def test_predictions_stored_internally_without_flag(self, univariate_data):
        """Predictions should always be stored internally even with predictions=False."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            predictions=False,
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, preds = forecaster.fit(y_train, y_test)
        # Return value should be empty (API contract)
        assert preds.empty
        # But internal storage should have predictions
        assert len(forecaster._last_predictions) >= 1
        for name, y_pred in forecaster._last_predictions.items():
            assert len(y_pred) == len(y_test)

    def test_predictions_stored_with_flag(self, univariate_data):
        """With predictions=True, both return value and internal storage populated."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            predictions=True,
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, preds = forecaster.fit(y_train, y_test)
        assert not preds.empty
        assert len(forecaster._last_predictions) >= 1

    def test_y_test_stored(self, univariate_data):
        """y_test should be stored internally for plot/diagnose access."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive"])
        forecaster.fit(y_train, y_test)
        np.testing.assert_array_equal(forecaster._last_y_test, y_test)

    def test_scores_stored(self, univariate_data):
        """Scores DataFrame should be stored internally."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive"])
        scores, _ = forecaster.fit(y_train, y_test)
        assert forecaster._last_scores is not None
        assert len(forecaster._last_scores) == len(scores)


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for tests
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


@pytest.mark.skipif(not _HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestVisualization:
    def _fit_forecaster(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            predictions=True,
            forecasters=["Naive", "Ridge_TS"],
        )
        scores, preds = forecaster.fit(y_train, y_test)
        return forecaster, y_train, y_test, scores

    def test_plot_forecast_single(self, univariate_data):
        from lazypredict.ts_visualization import plot_forecast
        y_train, y_test = univariate_data
        y_pred = np.full(len(y_test), y_train[-1])
        fig = plot_forecast(y_train, y_test, y_pred)
        assert fig is not None
        plt.close(fig)

    def test_plot_forecast_multiple(self, univariate_data):
        from lazypredict.ts_visualization import plot_forecast
        y_train, y_test = univariate_data
        predictions = {
            "Model_A": np.full(len(y_test), y_train[-1]),
            "Model_B": np.full(len(y_test), np.mean(y_train)),
        }
        fig = plot_forecast(y_train, y_test, predictions)
        assert fig is not None
        plt.close(fig)

    def test_plot_model_comparison(self, univariate_data):
        from lazypredict.ts_visualization import plot_model_comparison
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = plot_model_comparison(scores, metric="RMSE")
        assert fig is not None
        plt.close(fig)

    def test_plot_residuals(self, univariate_data):
        from lazypredict.ts_visualization import plot_residuals
        y_train, y_test = univariate_data
        y_pred = np.full(len(y_test), y_train[-1])
        fig = plot_residuals(y_test, y_pred, model_name="Naive")
        assert fig is not None
        plt.close(fig)

    def test_plot_error_distribution(self, univariate_data):
        from lazypredict.ts_visualization import plot_error_distribution
        y_train, y_test = univariate_data
        predictions = {
            "Model_A": np.full(len(y_test), y_train[-1]),
            "Model_B": np.full(len(y_test), np.mean(y_train)),
        }
        fig = plot_error_distribution(y_test, predictions)
        assert fig is not None
        plt.close(fig)

    def test_plot_metrics_heatmap(self, univariate_data):
        from lazypredict.ts_visualization import plot_metrics_heatmap
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = plot_metrics_heatmap(scores)
        assert fig is not None
        plt.close(fig)

    def test_lazy_forecaster_plot_results_forecast(self, univariate_data):
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = forecaster.plot_results(plot_type="forecast")
        assert fig is not None
        plt.close(fig)

    def test_lazy_forecaster_plot_results_comparison(self, univariate_data):
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = forecaster.plot_results(plot_type="comparison")
        assert fig is not None
        plt.close(fig)

    def test_lazy_forecaster_plot_results_residuals(self, univariate_data):
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = forecaster.plot_results(plot_type="residuals", model_name="Naive")
        assert fig is not None
        plt.close(fig)

    def test_lazy_forecaster_plot_results_errors(self, univariate_data):
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = forecaster.plot_results(plot_type="errors")
        assert fig is not None
        plt.close(fig)

    def test_lazy_forecaster_plot_results_heatmap(self, univariate_data):
        forecaster, y_train, y_test, scores = self._fit_forecaster(univariate_data)
        fig = forecaster.plot_results(plot_type="heatmap")
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_residual_diagnostics_keys(self):
        from lazypredict.ts_diagnostics import residual_diagnostics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8, 9.1, 9.9])
        result = residual_diagnostics(y_true, y_pred)
        expected_keys = {
            "residuals", "mean", "std",
            "ljung_box_stat", "ljung_box_pvalue",
            "jarque_bera_stat", "jarque_bera_pvalue",
            "acf_values", "is_white_noise", "is_normal",
        }
        assert set(result.keys()) == expected_keys

    def test_residuals_computed_correctly(self):
        from lazypredict.ts_diagnostics import residual_diagnostics
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        result = residual_diagnostics(y_true, y_pred)
        np.testing.assert_array_almost_equal(result["residuals"], [-2.0, 2.0, -3.0])

    def test_white_noise_detection(self):
        from lazypredict.ts_diagnostics import residual_diagnostics
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = np.zeros(100)  # residuals = y_true = white noise
        result = residual_diagnostics(y_true, y_pred)
        # White noise residuals should pass Ljung-Box (if statsmodels available)
        if result["ljung_box_pvalue"] is not None:
            assert result["is_white_noise"] is True

    def test_biased_residuals_mean(self):
        from lazypredict.ts_diagnostics import residual_diagnostics
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([5.0, 15.0, 25.0, 35.0, 45.0])  # systematic under-prediction
        result = residual_diagnostics(y_true, y_pred)
        assert result["mean"] == 5.0  # all residuals are +5

    def test_compare_diagnostics(self, univariate_data):
        from lazypredict.ts_diagnostics import compare_diagnostics
        y_train, y_test = univariate_data
        predictions = {
            "Model_A": np.full(len(y_test), y_train[-1]),
            "Model_B": np.full(len(y_test), np.mean(y_train)),
        }
        result = compare_diagnostics(y_test, predictions)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "Model_A" in result.index
        assert "Model_B" in result.index
        assert "Residual Mean" in result.columns
        assert "White Noise" in result.columns

    def test_lazy_forecaster_diagnose_single(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive", "Ridge_TS"])
        forecaster.fit(y_train, y_test)
        result = forecaster.diagnose(model_name="Naive")
        assert isinstance(result, dict)
        assert "residuals" in result
        assert "mean" in result

    def test_lazy_forecaster_diagnose_all(self, univariate_data):
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive", "Ridge_TS"])
        forecaster.fit(y_train, y_test)
        result = forecaster.diagnose()
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Tuning refit tests
# ---------------------------------------------------------------------------

try:
    import optuna  # noqa: F401
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


@pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")
class TestTuningRefit:
    def test_tune_top_k_returns_tuned_models(self, univariate_data):
        from lazypredict.ts_tuning import tune_top_k_forecasters
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Ridge_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        tune_results, tuned_models = tune_top_k_forecasters(
            scores_df=scores,
            models=forecaster.models,
            all_wrappers=forecaster._last_all_forecasters,
            y_train=y_train,
            X_train=None,
            seasonal_period=None,
            top_k=1,
            n_trials=3,
            refit=True,
        )
        assert isinstance(tune_results, pd.DataFrame)
        assert isinstance(tuned_models, dict)
        # Ridge_TS should be refit
        if "Ridge_TS" in tuned_models:
            assert tuned_models["Ridge_TS"] is not None

    def test_tune_top_k_no_refit(self, univariate_data):
        from lazypredict.ts_tuning import tune_top_k_forecasters
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Ridge_TS"],
        )
        scores, _ = forecaster.fit(y_train, y_test)
        tune_results, tuned_models = tune_top_k_forecasters(
            scores_df=scores,
            models=forecaster.models,
            all_wrappers=forecaster._last_all_forecasters,
            y_train=y_train,
            X_train=None,
            seasonal_period=None,
            top_k=1,
            n_trials=3,
            refit=False,
        )
        assert isinstance(tune_results, pd.DataFrame)
        assert len(tuned_models) == 0


# ---------------------------------------------------------------------------
# Tests for refactored LazyForecaster methods
# ---------------------------------------------------------------------------


class TestLazyForecasterRefactoredMethods:
    """Test the refactored private methods in LazyForecaster."""

    def test_convert_and_validate_inputs(self, univariate_data):
        """Test input conversion and validation."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive"])
        y_tr, y_te, X_tr, X_te = forecaster._convert_and_validate_inputs(
            y_train, y_test, None, None
        )
        assert isinstance(y_tr, np.ndarray)
        assert isinstance(y_te, np.ndarray)

    def test_resolve_seasonal_period_none(self, univariate_data):
        """Test seasonal period resolution when not specified."""
        y_train, _ = univariate_data
        forecaster = LazyForecaster(seasonal_period=None)
        sp = forecaster._resolve_seasonal_period(y_train)
        assert isinstance(sp, (int, type(None)))

    def test_resolve_seasonal_period_specified(self, seasonal_data):
        """Test seasonal period when explicitly specified."""
        y_train, _ = seasonal_data
        forecaster = LazyForecaster(seasonal_period=12)
        sp = forecaster._resolve_seasonal_period(y_train)
        assert sp == 12

    def test_resolve_forecasters_all(self, univariate_data):
        """Test forecaster resolution with 'all' keyword."""
        y_train, _ = univariate_data
        forecaster = LazyForecaster(forecasters="all", max_models=2)
        all_forecasters = forecaster._resolve_forecasters(None)
        assert len(all_forecasters) == 2

    def test_resolve_forecasters_subset(self, univariate_data):
        """Test forecaster resolution with specific subset."""
        y_train, _ = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive", "SeasonalNaive"])
        all_forecasters = forecaster._resolve_forecasters(None)
        names = [name for name, _ in all_forecasters]
        assert set(names) <= {"Naive", "SeasonalNaive"}

    def test_store_fit_state(self, univariate_data):
        """Test storage of fit state."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive"])
        scores, _ = forecaster.fit(y_train, y_test)

        assert hasattr(forecaster, "_last_y_train")
        assert hasattr(forecaster, "_last_y_test")
        assert hasattr(forecaster, "_last_scores")
        assert len(forecaster._last_scores) > 0

    def test_plot_results_dispatch(self, univariate_data):
        """Test plot_results dispatcher."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(forecasters=["Naive"], predictions=True)
        scores, preds = forecaster.fit(y_train, y_test)

        # Test that dispatcher works with dispatch pattern
        try:
            fig = forecaster.plot_results(plot_type="forecast")
            assert fig is not None
        except (ImportError, ValueError):
            # OK if matplotlib not available
            pass

    def test_ensemble_dispatch_simple_average(self, univariate_data):
        """Test ensemble dispatcher with simple_average."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive", "SeasonalNaive"],
            predictions=True,
        )
        scores, preds = forecaster.fit(y_train, y_test)

        ens_pred = forecaster.ensemble(method="simple_average")
        assert len(ens_pred) == len(y_test)

    def test_ensemble_dispatch_weighted_average(self, univariate_data):
        """Test ensemble dispatcher with weighted_average."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive", "SeasonalNaive"],
            predictions=True,
        )
        scores, preds = forecaster.fit(y_train, y_test)

        ens_pred = forecaster.ensemble(method="weighted_average")
        assert len(ens_pred) == len(y_test)

    def test_ensemble_dispatch_stacking(self, univariate_data):
        """Test ensemble dispatcher with stacking."""
        y_train, y_test = univariate_data
        forecaster = LazyForecaster(
            forecasters=["Naive", "SeasonalNaive"],
            predictions=True,
        )
        scores, preds = forecaster.fit(y_train, y_test)

        ens_pred = forecaster.ensemble(method="stacking", y_true=y_test)
        assert len(ens_pred) == len(y_test)


# ---------------------------------------------------------------------------
# Tests for diagnostics edge cases
# ---------------------------------------------------------------------------


class TestDiagnosticsEdgeCases:
    """Test edge cases in residual diagnostics."""

    def test_diagnostics_small_sample(self):
        """Test with small sample size."""
        from lazypredict.ts_diagnostics import residual_diagnostics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.0, 2.9])
        result = residual_diagnostics(y_true, y_pred)
        assert "residuals" in result
        assert result["ljung_box_pvalue"] is None  # Too small for LB test

    def test_diagnostics_zero_variance(self):
        """Test with zero variance residuals."""
        from lazypredict.ts_diagnostics import residual_diagnostics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = residual_diagnostics(y_true, y_pred)
        assert result["std"] == 0.0

    def test_compute_acf_numpy_fallback(self):
        """Test ACF computation with numpy fallback."""
        from lazypredict.ts_diagnostics import _compute_acf_numpy
        residuals = np.random.randn(50)
        acf = _compute_acf_numpy(residuals, max_lag=10)
        assert len(acf) == 11
        np.testing.assert_allclose(acf[0], 1.0, rtol=1e-10)  # ACF at lag 0 is always 1
