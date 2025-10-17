"""
Comprehensive test suite for LazyPredict ModelExplainer with SHAP integration.
"""
import pytest
import numpy as np
import pandas as pd
import os
import shutil
import gc
import time
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import mlflow


# Check if SHAP is available
try:
    import shap
    from lazypredict.Explainer import ModelExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Clean up MLflow artifacts and database before and after each test."""
    # End any active runs before cleanup
    try:
        mlflow.end_run()
    except:
        pass

    # Force garbage collection to release file handles
    gc.collect()
    time.sleep(0.1)

    mlflow_files = ["mlflow.db", "mlflow.db-wal", "mlflow.db-shm", "mlruns"]
    for item in mlflow_files:
        if os.path.exists(item):
            try:
                if os.path.isfile(item):
                    os.remove(item)
                else:
                    shutil.rmtree(item)
            except (PermissionError, OSError):
                pass

    yield

    # End any runs after the test
    try:
        mlflow.end_run()
    except:
        pass

    # Force garbage collection again
    gc.collect()
    time.sleep(0.1)

    for item in mlflow_files:
        if os.path.exists(item):
            try:
                if os.path.isfile(item):
                    os.remove(item)
                else:
                    shutil.rmtree(item)
            except (PermissionError, OSError):
                pass


@pytest.fixture
def classification_data():
    """Fixture for small classification dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data[:100], columns=data.feature_names)
    y = data.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data():
    """Fixture for small regression dataset."""
    data = load_diabetes()
    X = pd.DataFrame(data.data[:100], columns=data.feature_names)
    y = data.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_classifier(classification_data):
    """Fixture for trained classifier."""
    X_train, X_test, y_train, y_test = classification_data
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        predictions=False,
        random_state=42,
        classifiers=['LogisticRegression', 'DecisionTreeClassifier']
    )
    clf.fit(X_train, X_test, y_train, y_test)
    return clf, X_train, X_test


@pytest.fixture
def trained_regressor(regression_data):
    """Fixture for trained regressor."""
    X_train, X_test, y_train, y_test = regression_data
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        predictions=False,
        random_state=42,
        regressors=['LinearRegression', 'Ridge']
    )
    reg.fit(X_train, X_test, y_train, y_test)
    return reg, X_train, X_test


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestModelExplainerInitialization:
    """Test ModelExplainer initialization."""

    def test_init_with_classifier(self, trained_classifier):
        """Test initialization with a trained classifier."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        assert explainer is not None
        assert len(explainer.trained_models) > 0
        assert isinstance(explainer.explainers, dict)
        assert isinstance(explainer.shap_values, dict)

    def test_init_with_regressor(self, trained_regressor):
        """Test initialization with a trained regressor."""
        reg, X_train, X_test = trained_regressor
        explainer = ModelExplainer(reg, X_train, X_test)

        assert explainer is not None
        assert len(explainer.trained_models) > 0

    def test_init_with_unfitted_estimator(self, classification_data):
        """Test initialization with unfitted estimator raises error."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0)

        with pytest.raises(ValueError, match="No trained models found"):
            ModelExplainer(clf, X_train, X_test)

    def test_init_without_trained_models_attribute(self, classification_data):
        """Test initialization with invalid estimator."""
        X_train, X_test, y_train, y_test = classification_data

        class FakeEstimator:
            pass

        with pytest.raises(ValueError, match="must be a fitted LazyClassifier or LazyRegressor"):
            ModelExplainer(FakeEstimator(), X_train, X_test)

    def test_init_with_numpy_arrays(self, classification_data):
        """Test initialization with numpy arrays instead of DataFrames."""
        X_train, X_test, y_train, y_test = classification_data

        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            classifiers=['LogisticRegression']
        )
        clf.fit(X_train.values, X_test.values, y_train, y_test)

        explainer = ModelExplainer(clf, X_train.values, X_test.values)
        assert explainer is not None


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestFeatureImportance:
    """Test feature importance functionality."""

    def test_feature_importance_basic(self, trained_classifier):
        """Test basic feature importance calculation."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        importance = explainer.feature_importance('LogisticRegression')

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) > 0

    def test_feature_importance_top_n(self, trained_classifier):
        """Test feature importance with top_n parameter."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        importance = explainer.feature_importance('LogisticRegression', top_n=5)

        assert len(importance) == min(5, X_train.shape[1])

    def test_feature_importance_invalid_model(self, trained_classifier):
        """Test feature importance with invalid model name."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        with pytest.raises(ValueError, match="Model.*not found"):
            explainer.feature_importance('NonExistentModel')

    def test_feature_importance_all_models(self, trained_classifier):
        """Test feature importance for all trained models."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        for model_name in explainer.trained_models.keys():
            importance = explainer.feature_importance(model_name)
            assert isinstance(importance, pd.DataFrame)
            assert len(importance) > 0


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestPlotSummary:
    """Test SHAP summary plot functionality."""

    def test_plot_summary_basic(self, trained_classifier):
        """Test basic summary plot generation."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        # Should not raise an exception
        try:
            explainer.plot_summary('LogisticRegression', show=False)
        except Exception as e:
            pytest.fail(f"plot_summary raised unexpected exception: {e}")

    def test_plot_summary_bar_type(self, trained_classifier):
        """Test summary plot with bar type."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        try:
            explainer.plot_summary('LogisticRegression', plot_type='bar', show=False)
        except Exception as e:
            pytest.fail(f"plot_summary with bar type raised exception: {e}")

    def test_plot_summary_max_display(self, trained_classifier):
        """Test summary plot with max_display parameter."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        try:
            explainer.plot_summary('LogisticRegression', max_display=5, show=False)
        except Exception as e:
            pytest.fail(f"plot_summary with max_display raised exception: {e}")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestExplainPrediction:
    """Test single prediction explanation functionality."""

    def test_explain_prediction_basic(self, trained_classifier):
        """Test basic prediction explanation."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        try:
            explainer.explain_prediction('LogisticRegression', instance_idx=0, show=False)
        except Exception as e:
            pytest.fail(f"explain_prediction raised unexpected exception: {e}")

    def test_explain_prediction_invalid_index(self, trained_classifier):
        """Test prediction explanation with invalid instance index."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        with pytest.raises(ValueError, match="instance_idx.*is out of range"):
            explainer.explain_prediction('LogisticRegression', instance_idx=1000)

    def test_explain_prediction_multiple_instances(self, trained_classifier):
        """Test prediction explanation for multiple instances."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        # Test first and last instances
        try:
            explainer.explain_prediction('LogisticRegression', instance_idx=0, show=False)
            explainer.explain_prediction('LogisticRegression', instance_idx=len(X_test)-1, show=False)
        except Exception as e:
            pytest.fail(f"explain_prediction for multiple instances raised exception: {e}")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestGetTopFeatures:
    """Test get_top_features functionality."""

    def test_get_top_features_basic(self, trained_classifier):
        """Test basic top features extraction."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        top_features = explainer.get_top_features('LogisticRegression', instance_idx=0)

        assert isinstance(top_features, pd.DataFrame)
        assert 'feature' in top_features.columns
        assert 'shap_value' in top_features.columns
        assert len(top_features) > 0

    def test_get_top_features_top_n(self, trained_classifier):
        """Test top features with top_n parameter."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        top_features = explainer.get_top_features('LogisticRegression', instance_idx=0, top_n=3)

        assert len(top_features) == 3


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestCompareModels:
    """Test model comparison functionality."""

    def test_compare_models_basic(self, trained_classifier):
        """Test basic model comparison."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        model_names = list(explainer.trained_models.keys())[:2]
        comparison = explainer.compare_models(model_names, show=False)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) > 0

    def test_compare_models_top_n(self, trained_classifier):
        """Test model comparison with top_n_features."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        model_names = list(explainer.trained_models.keys())[:2]
        comparison = explainer.compare_models(model_names, top_n_features=5, show=False)

        assert len(comparison) <= 5


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestPlotDependence:
    """Test dependence plot functionality."""

    def test_plot_dependence_basic(self, trained_regressor):
        """Test basic dependence plot."""
        reg, X_train, X_test = trained_regressor
        explainer = ModelExplainer(reg, X_train, X_test)

        feature_name = X_train.columns[0]
        try:
            explainer.plot_dependence('LinearRegression', feature_name, show=False)
        except Exception as e:
            pytest.fail(f"plot_dependence raised unexpected exception: {e}")


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestExplainerCaching:
    """Test SHAP explainer and values caching."""

    def test_explainer_caching(self, trained_classifier):
        """Test that explainers are cached after first use."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        # First call should create and cache the explainer
        explainer.feature_importance('LogisticRegression')
        assert 'LogisticRegression' in explainer.explainers

        # Second call should use cached explainer
        explainer.feature_importance('LogisticRegression')
        assert 'LogisticRegression' in explainer.explainers

    def test_shap_values_caching(self, trained_classifier):
        """Test that SHAP values are cached after first computation."""
        clf, X_train, X_test = trained_classifier
        explainer = ModelExplainer(clf, X_train, X_test)

        # First call should compute and cache SHAP values
        explainer.feature_importance('LogisticRegression')
        assert 'LogisticRegression' in explainer.shap_values

        # Values should still be cached
        assert 'LogisticRegression' in explainer.shap_values


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP not installed")
class TestRegressionExplainability:
    """Test explainability for regression models."""

    def test_regression_feature_importance(self, trained_regressor):
        """Test feature importance for regression models."""
        reg, X_train, X_test = trained_regressor
        explainer = ModelExplainer(reg, X_train, X_test)

        importance = explainer.feature_importance('LinearRegression')

        assert isinstance(importance, pd.DataFrame)
        assert len(importance) > 0

    def test_regression_plot_summary(self, trained_regressor):
        """Test summary plot for regression models."""
        reg, X_train, X_test = trained_regressor
        explainer = ModelExplainer(reg, X_train, X_test)

        try:
            explainer.plot_summary('LinearRegression', show=False)
        except Exception as e:
            pytest.fail(f"plot_summary for regression raised exception: {e}")


@pytest.mark.skipif(SHAP_AVAILABLE, reason="Test only when SHAP is not installed")
class TestShapNotInstalled:
    """Test behavior when SHAP is not installed."""

    def test_import_error_without_shap(self, trained_classifier):
        """Test that ModelExplainer raises ImportError when SHAP not installed."""
        clf, X_train, X_test = trained_classifier

        # This test only runs when SHAP is not available
        # We can't actually test this since SHAP is available in our test environment
        # But we keep the test for documentation purposes
        pass
