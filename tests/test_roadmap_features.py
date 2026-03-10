"""Tests for all roadmap features: explainability, tuning, forecasting optimization,
auto-convert, ensemble, horizon, EBM, Dask, Spark, FLAML.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split


# ============================================================================
# Phase 1: Explainability
# ============================================================================


class TestExplainPermutation:
    """Test permutation importance explain()."""

    @pytest.fixture
    def clf_data(self):
        data = load_breast_cancer()
        X = pd.DataFrame(data.data[:, :5], columns=[f"f{i}" for i in range(5)])
        y = data.target
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.fixture
    def reg_data(self):
        data = load_diabetes()
        X = pd.DataFrame(data.data[:, :5], columns=[f"f{i}" for i in range(5)])
        y = data.target
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def test_classifier_explain_permutation(self, clf_data):
        from lazypredict.Supervised import LazyClassifier

        X_train, X_test, y_train, y_test = clf_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True, max_models=3)
        scores, _ = clf.fit(X_train, X_test, y_train, y_test)
        assert len(clf.models) > 0

        importance_df = clf.explain(X_test, y_test, method="permutation", n_repeats=3)
        assert isinstance(importance_df, pd.DataFrame)
        assert importance_df.shape[0] == X_test.shape[1]
        assert importance_df.shape[1] == len(clf.models)

    def test_regressor_explain_permutation(self, reg_data):
        from lazypredict.Supervised import LazyRegressor

        X_train, X_test, y_train, y_test = reg_data
        reg = LazyRegressor(verbose=0, ignore_warnings=True, max_models=3)
        scores, _ = reg.fit(X_train, X_test, y_train, y_test)

        importance_df = reg.explain(X_test, y_test, method="permutation", n_repeats=3)
        assert isinstance(importance_df, pd.DataFrame)
        assert importance_df.shape[0] == X_test.shape[1]

    def test_explain_no_models_raises(self, clf_data):
        from lazypredict.Supervised import LazyClassifier

        X_train, X_test, y_train, y_test = clf_data
        clf = LazyClassifier()
        with pytest.raises(ValueError, match="No models fitted"):
            clf.explain(X_test, y_test)

    def test_explain_invalid_method(self, clf_data):
        from lazypredict.Supervised import LazyClassifier

        X_train, X_test, y_train, y_test = clf_data
        clf = LazyClassifier(verbose=0, max_models=2)
        clf.fit(X_train, X_test, y_train, y_test)
        with pytest.raises(ValueError, match="Unknown method"):
            clf.explain(X_test, y_test, method="invalid")


# ============================================================================
# Phase 1: Auto-convert PySpark/Dask DataFrames
# ============================================================================


class TestAutoConvert:
    """Test auto-conversion of PySpark/Dask inputs."""

    def test_pandas_passthrough(self):
        from lazypredict.distributed import auto_convert_dataframe

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = auto_convert_dataframe(df, "test")
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_numpy_passthrough(self):
        from lazypredict.distributed import auto_convert_dataframe

        arr = np.array([1, 2, 3])
        result = auto_convert_dataframe(arr, "test")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)


# ============================================================================
# Phase 2: Search Spaces
# ============================================================================


class TestSearchSpaces:
    """Test search space registry."""

    def test_supervised_spaces_exist(self):
        from lazypredict.search_spaces import SEARCH_SPACES, get_search_space

        assert len(SEARCH_SPACES) > 20
        assert get_search_space("RandomForestClassifier") is not None
        assert get_search_space("XGBRegressor") is not None
        assert get_search_space("DummyClassifier") is None

    def test_ts_spaces_exist(self):
        from lazypredict.ts_search_spaces import TS_SEARCH_SPACES, get_ts_search_space

        assert len(TS_SEARCH_SPACES) > 15
        assert get_ts_search_space("SARIMAX") is not None
        assert get_ts_search_space("RandomForestRegressor_TS") is not None
        assert get_ts_search_space("LSTM_TS") is not None
        assert get_ts_search_space("Naive") is None


# ============================================================================
# Phase 2: Tuning parameters accepted
# ============================================================================


class TestTuneParameters:
    """Test that tune parameters are accepted by constructors."""

    def test_classifier_tune_params(self):
        from lazypredict.Supervised import LazyClassifier

        clf = LazyClassifier(
            tune=True, tune_top_k=3, tune_trials=10,
            tune_timeout=30, tune_backend="optuna",
        )
        assert clf.tune is True
        assert clf.tune_top_k == 3
        assert clf.tune_trials == 10
        assert clf.tune_backend == "optuna"

    def test_regressor_tune_params(self):
        from lazypredict.Supervised import LazyRegressor

        reg = LazyRegressor(
            tune=True, tune_top_k=5, tune_backend="sklearn",
        )
        assert reg.tune is True
        assert reg.tune_backend == "sklearn"

    def test_invalid_tune_backend(self):
        from lazypredict.Supervised import LazyClassifier

        with pytest.raises(ValueError, match="tune_backend"):
            LazyClassifier(tune_backend="invalid")

    def test_forecaster_tune_params(self):
        from lazypredict.TimeSeriesForecasting import LazyForecaster

        fc = LazyForecaster(
            tune=True, tune_top_k=3, tune_trials=20,
            tune_metric="MAE", tune_seasonal=True,
            horizon_strategy="direct",
        )
        assert fc.tune is True
        assert fc.tune_metric == "MAE"
        assert fc.tune_seasonal is True
        assert fc.horizon_strategy == "direct"

    def test_invalid_tune_metric(self):
        from lazypredict.TimeSeriesForecasting import LazyForecaster

        with pytest.raises(ValueError, match="tune_metric"):
            LazyForecaster(tune_metric="invalid")

    def test_invalid_horizon_strategy(self):
        from lazypredict.TimeSeriesForecasting import LazyForecaster

        with pytest.raises(ValueError, match="horizon_strategy"):
            LazyForecaster(horizon_strategy="invalid")


# ============================================================================
# Phase 3: Ensemble
# ============================================================================


class TestEnsemble:
    """Test ensemble methods."""

    def test_simple_average(self):
        from lazypredict.ensemble import ensemble_simple_average

        preds = {
            "model_a": np.array([1.0, 2.0, 3.0]),
            "model_b": np.array([2.0, 3.0, 4.0]),
        }
        result = ensemble_simple_average(preds)
        np.testing.assert_array_almost_equal(result, [1.5, 2.5, 3.5])

    def test_weighted_average(self):
        from lazypredict.ensemble import ensemble_weighted_average

        preds = {
            "model_a": np.array([1.0, 2.0, 3.0]),
            "model_b": np.array([2.0, 3.0, 4.0]),
        }
        scores = {"model_a": 0.5, "model_b": 1.5}  # model_a has lower error
        result = ensemble_weighted_average(preds, scores)
        # model_a should have higher weight (lower error)
        assert result[0] < 1.5  # closer to model_a

    def test_stacking(self):
        from lazypredict.ensemble import ensemble_stacking

        preds = {
            "model_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "model_b": np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        }
        y_true = np.array([1.2, 2.1, 3.3, 4.1, 5.2])
        result = ensemble_stacking(preds, y_true)
        assert len(result) == 5

    def test_empty_predictions_raises(self):
        from lazypredict.ensemble import ensemble_simple_average

        with pytest.raises(ValueError, match="No predictions"):
            ensemble_simple_average({})


# ============================================================================
# Phase 3: Horizon strategies
# ============================================================================


class TestHorizonStrategies:
    """Test direct and multi-output forecasting."""

    @pytest.fixture
    def ts_data(self):
        np.random.seed(42)
        return np.cumsum(np.random.randn(100)) + 100

    def test_direct_forecast(self, ts_data):
        from sklearn.linear_model import Ridge

        from lazypredict.horizon import direct_forecast

        preds = direct_forecast(
            Ridge, ts_data, horizon=5, n_lags=5, n_rolling=(3,),
        )
        assert len(preds) == 5
        assert not np.any(np.isnan(preds))

    def test_multi_output_forecast(self, ts_data):
        from sklearn.linear_model import Ridge

        from lazypredict.horizon import multi_output_forecast

        preds = multi_output_forecast(
            Ridge, ts_data, horizon=5, n_lags=5, n_rolling=(3,),
        )
        assert len(preds) == 5
        assert not np.any(np.isnan(preds))


# ============================================================================
# Phase 3: Dask backend
# ============================================================================


class TestDaskBackend:
    """Test Dask joblib backend registration."""

    def test_register_without_dask(self):
        """Should return False if dask not installed (or True if it is)."""
        from lazypredict.distributed import register_dask_backend

        # Just test it doesn't crash
        result = register_dask_backend()
        assert isinstance(result, bool)


# ============================================================================
# Phase 3: EBM model integration
# ============================================================================


class TestEBMIntegration:
    """Test InterpretML EBM model availability."""

    def test_ebm_in_model_lists(self):
        """If interpret is installed, EBM should be in model lists."""
        try:
            import interpret  # noqa: F401

            from lazypredict.Supervised import CLASSIFIERS, REGRESSORS

            clf_names = [name for name, _ in CLASSIFIERS]
            reg_names = [name for name, _ in REGRESSORS]
            assert "ExplainableBoostingClassifier" in clf_names
            assert "ExplainableBoostingRegressor" in reg_names
        except ImportError:
            pytest.skip("interpret not installed")


# ============================================================================
# Phase 4: Spark MLlib
# ============================================================================


class TestSparkMLlib:
    """Test Spark MLlib classes exist and validate."""

    def test_spark_classes_importable(self):
        from lazypredict.spark import LazySparkClassifier, LazySparkRegressor

        # Classes exist even without pyspark
        assert LazySparkClassifier is not None
        assert LazySparkRegressor is not None

    def test_spark_requires_pyspark(self):
        """Without pyspark, instantiation should raise ImportError."""
        try:
            import pyspark  # noqa: F401

            pytest.skip("PySpark is installed")
        except ImportError:
            from lazypredict.spark import LazySparkClassifier

            with pytest.raises(ImportError, match="PySpark"):
                LazySparkClassifier()


# ============================================================================
# Integration: Full flow with tune=False (basic sanity)
# ============================================================================


class TestFullFlow:
    """Sanity test that the full pipeline still works with new params."""

    def test_classifier_basic_flow(self):
        from lazypredict.Supervised import LazyClassifier

        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data[:, :5], data.target, test_size=0.3, random_state=42,
        )
        clf = LazyClassifier(verbose=0, max_models=3, tune=False)
        scores, preds = clf.fit(X_train, X_test, y_train, y_test)
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) > 0

    def test_regressor_basic_flow(self):
        from lazypredict.Supervised import LazyRegressor

        data = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data[:, :5], data.target, test_size=0.3, random_state=42,
        )
        reg = LazyRegressor(verbose=0, max_models=3, tune=False)
        scores, preds = reg.fit(X_train, X_test, y_train, y_test)
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) > 0

    def test_forecaster_basic_flow(self):
        from lazypredict.TimeSeriesForecasting import LazyForecaster

        np.random.seed(42)
        y = np.cumsum(np.random.randn(120)) + 100
        y_train, y_test = y[:100], y[100:]

        fc = LazyForecaster(
            verbose=0, max_models=3, tune=False,
            forecasters=["Naive", "SeasonalNaive", "LinearRegression_TS"],
        )
        scores, preds = fc.fit(y_train, y_test)
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) > 0


# ============================================================================
# Module imports
# ============================================================================


class TestModuleImports:
    """Test that all new modules are importable."""

    def test_explainability(self):
        from lazypredict.explainability import explain_permutation

        assert callable(explain_permutation)

    def test_search_spaces(self):
        from lazypredict.search_spaces import SEARCH_SPACES

        assert isinstance(SEARCH_SPACES, dict)

    def test_ts_search_spaces(self):
        from lazypredict.ts_search_spaces import TS_SEARCH_SPACES

        assert isinstance(TS_SEARCH_SPACES, dict)

    def test_tuning(self):
        from lazypredict.tuning import tune_top_k

        assert callable(tune_top_k)

    def test_ts_tuning(self):
        from lazypredict.ts_tuning import tune_top_k_forecasters

        assert callable(tune_top_k_forecasters)

    def test_ensemble(self):
        from lazypredict.ensemble import (
            ensemble_simple_average,
            ensemble_stacking,
            ensemble_weighted_average,
        )

        assert callable(ensemble_simple_average)
        assert callable(ensemble_weighted_average)
        assert callable(ensemble_stacking)

    def test_horizon(self):
        from lazypredict.horizon import direct_forecast, multi_output_forecast

        assert callable(direct_forecast)
        assert callable(multi_output_forecast)

    def test_distributed(self):
        from lazypredict.distributed import auto_convert_dataframe

        assert callable(auto_convert_dataframe)

    def test_spark(self):
        from lazypredict.spark import LazySparkClassifier, LazySparkRegressor

        assert LazySparkClassifier is not None
        assert LazySparkRegressor is not None
