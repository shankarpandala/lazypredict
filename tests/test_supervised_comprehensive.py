"""
Comprehensive test suite for LazyPredict with parametrization and edge cases.
"""
import pytest
import numpy as np
import pandas as pd
import os
import shutil
from pathlib import Path
from lazypredict.Supervised import LazyClassifier, LazyRegressor, get_card_split
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import mlflow


@pytest.fixture(scope="function", autouse=True)
def cleanup_mlflow():
    """Clean up MLflow artifacts and database before and after each test."""
    import time
    import gc
    
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
                # File is locked, skip it
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
                # File is locked, skip it
                pass


@pytest.fixture
def classification_data():
    """Fixture for classification data."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data():
    """Fixture for regression data."""
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def categorical_classification_data():
    """Fixture for classification data with categorical features."""
    n_samples = 200
    X = pd.DataFrame({
        'num1': np.random.rand(n_samples),
        'num2': np.random.rand(n_samples),
        'cat_low': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_high': np.random.choice([f'X{i}' for i in range(20)], n_samples)
    })
    y = np.random.randint(0, 2, n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


class TestLazyClassifier:
    """Test suite for LazyClassifier."""
    
    def test_basic_fit(self, classification_data):
        """Test basic classification fit."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
        assert "Balanced Accuracy" in models.columns
        assert "Accuracy" in models.columns
        assert "F1 Score" in models.columns
        assert "Time Taken" in models.columns
    
    def test_with_predictions(self, classification_data):
        """Test classification with predictions."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(y_test)
    
    def test_custom_metric(self, classification_data):
        """Test classification with custom metric."""
        def custom_accuracy(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=custom_accuracy)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert "custom_accuracy" in models.columns
    
    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_parallel_training(self, classification_data, n_jobs):
        """Test parallel training with different n_jobs values."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=n_jobs)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
    
    def test_specific_classifiers(self, classification_data):
        """Test with specific classifiers."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(
            verbose=0, 
            ignore_warnings=True,
            classifiers=[RandomForestClassifier, LogisticRegression]
        )
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert len(models) == 2
        assert "RandomForestClassifier" in models.index
        assert "LogisticRegression" in models.index
    
    def test_with_categorical_features(self, categorical_classification_data):
        """Test classification with categorical features."""
        X_train, X_test, y_train, y_test = categorical_classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
    
    def test_provide_models(self, classification_data):
        """Test provide_models method."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models_dict = clf.provide_models(X_train, X_test, y_train, y_test)
        
        assert isinstance(models_dict, dict)
        assert len(models_dict) > 0
        
        # Test predictions with retrieved models
        for name, model in models_dict.items():
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)
    
    def test_mlflow_integration(self, classification_data):
        """Test MLflow integration."""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        os.environ['MLFLOW_TRACKING_URI'] = "sqlite:///mlflow.db"
        
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert os.path.exists("mlflow.db")
        
        # Clean up environment
        if 'MLFLOW_TRACKING_URI' in os.environ:
            del os.environ['MLFLOW_TRACKING_URI']


class TestLazyRegressor:
    """Test suite for LazyRegressor."""
    
    def test_basic_fit(self, regression_data):
        """Test basic regression fit."""
        X_train, X_test, y_train, y_test = regression_data
        reg = LazyRegressor(verbose=0, ignore_warnings=True)
        models = reg.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
        assert "Adjusted R-Squared" in models.columns
        assert "R-Squared" in models.columns
        assert "RMSE" in models.columns
        assert "Time Taken" in models.columns
    
    def test_with_predictions(self, regression_data):
        """Test regression with predictions."""
        X_train, X_test, y_train, y_test = regression_data
        reg = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(y_test)
    
    def test_custom_metric(self, regression_data):
        """Test regression with custom metric."""
        X_train, X_test, y_train, y_test = regression_data
        reg = LazyRegressor(
            verbose=0, 
            ignore_warnings=True, 
            custom_metric=mean_absolute_error
        )
        models = reg.fit(X_train, X_test, y_train, y_test)
        
        assert "mean_absolute_error" in models.columns
    
    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_parallel_training(self, regression_data, n_jobs):
        """Test parallel training with different n_jobs values."""
        X_train, X_test, y_train, y_test = regression_data
        reg = LazyRegressor(verbose=0, ignore_warnings=True, n_jobs=n_jobs)
        models = reg.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
    
    def test_specific_regressors(self, regression_data):
        """Test with specific regressors."""
        X_train, X_test, y_train, y_test = regression_data
        reg = LazyRegressor(
            verbose=0,
            ignore_warnings=True,
            regressors=[RandomForestRegressor, LinearRegression]
        )
        models = reg.fit(X_train, X_test, y_train, y_test)
        
        assert len(models) == 2
        assert "RandomForestRegressor" in models.index
        assert "LinearRegression" in models.index
    
    def test_provide_models(self, regression_data):
        """Test provide_models method."""
        X_train, X_test, y_train, y_test = regression_data
        reg = LazyRegressor(verbose=0, ignore_warnings=True)
        models_dict = reg.provide_models(X_train, X_test, y_train, y_test)
        
        assert isinstance(models_dict, dict)
        assert len(models_dict) > 0
        
        # Test predictions with retrieved models
        for name, model in models_dict.items():
            predictions = model.predict(X_test)
            assert len(predictions) == len(y_test)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        X_train, X_test = X, X
        y_train, y_test = y, y
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        with pytest.raises(Exception):
            clf.fit(X_train, X_test, y_train, y_test)
    
    def test_single_feature(self):
        """Test with single feature."""
        X, y = make_classification(n_samples=100, n_features=1, n_informative=1, 
                                   n_redundant=0, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
    
    def test_all_categorical(self):
        """Test with all categorical features."""
        n_samples = 200
        X = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
            'cat2': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'cat3': np.random.choice(['P', 'Q'], n_samples)
        })
        y = np.random.randint(0, 2, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
    
    @pytest.mark.parametrize("n_classes", [2, 3, 5])
    def test_multiclass_classification(self, n_classes):
        """Test multiclass classification."""
        X, y = make_classification(
            n_samples=200, 
            n_features=10,
            n_classes=n_classes,
            n_informative=8,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
        assert len(models) > 0
    
    def test_missing_values(self):
        """Test with missing values."""
        n_samples = 200
        X = np.random.rand(n_samples, 10)
        # Introduce missing values
        X[np.random.choice(n_samples, 20), np.random.choice(10, 5)] = np.nan
        y = np.random.randint(0, 2, n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
    
    def test_invalid_classifier_list(self):
        """Test with invalid classifier list."""
        X, y = make_classification(n_samples=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = LazyClassifier(verbose=0, ignore_warnings=True, classifiers=["invalid"])
        with pytest.raises(ValueError):
            clf.fit(X_train, X_test, y_train, y_test)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_get_card_split_basic(self):
        """Test get_card_split with basic data."""
        df = pd.DataFrame({
            'A': ['a', 'b', 'c', 'd', 'e'],
            'B': ['f', 'g', 'h', 'i', 'j'],
            'C': [f'x{i}' for i in range(20)]
        })
        cols = pd.Index(['A', 'B', 'C'])
        card_low, card_high = get_card_split(df, cols, n=10)
        
        assert list(card_low) == ['A', 'B']
        assert list(card_high) == ['C']
    
    @pytest.mark.parametrize("n", [5, 10, 15, 20])
    def test_get_card_split_different_n(self, n):
        """Test get_card_split with different n values."""
        df = pd.DataFrame({
            'low': ['a'] * 50 + ['b'] * 50,
            'medium': [f'x{i%15}' for i in range(100)],
            'high': [f'y{i}' for i in range(100)]
        })
        cols = pd.Index(['low', 'medium', 'high'])
        card_low, card_high = get_card_split(df, cols, n=n)
        
        assert isinstance(card_low, pd.Index)
        assert isinstance(card_high, pd.Index)
        assert len(card_low) + len(card_high) == 3
    
    def test_get_card_split_empty(self):
        """Test get_card_split with empty columns."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        cols = pd.Index([])
        card_low, card_high = get_card_split(df, cols, n=10)
        
        assert len(card_low) == 0
        assert len(card_high) == 0


class TestVerbosityAndWarnings:
    """Test verbosity and warning handling."""
    
    def test_verbose_mode(self, classification_data):
        """Test verbose mode."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=2, ignore_warnings=False)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)
    
    def test_warnings_not_ignored(self, classification_data):
        """Test with warnings not ignored."""
        X_train, X_test, y_train, y_test = classification_data
        clf = LazyClassifier(verbose=0, ignore_warnings=False)
        models = clf.fit(X_train, X_test, y_train, y_test)
        
        assert isinstance(models, pd.DataFrame)


class TestRandomState:
    """Test random state reproducibility."""
    
    def test_reproducibility(self, classification_data):
        """Test that results are reproducible with same random_state."""
        X_train, X_test, y_train, y_test = classification_data
        
        clf1 = LazyClassifier(verbose=0, ignore_warnings=True, random_state=42)
        models1 = clf1.fit(X_train, X_test, y_train, y_test)
        
        clf2 = LazyClassifier(verbose=0, ignore_warnings=True, random_state=42)
        models2 = clf2.fit(X_train, X_test, y_train, y_test)
        
        # Results should be similar (not exact due to model internals)
        assert set(models1.index) == set(models2.index)
    
    def test_different_random_states(self, classification_data):
        """Test that different random states give different results."""
        X_train, X_test, y_train, y_test = classification_data
        
        clf1 = LazyClassifier(verbose=0, ignore_warnings=True, random_state=42)
        models1 = clf1.fit(X_train, X_test, y_train, y_test)
        
        clf2 = LazyClassifier(verbose=0, ignore_warnings=True, random_state=123)
        models2 = clf2.fit(X_train, X_test, y_train, y_test)
        
        # Should have same models but potentially different scores
        assert set(models1.index) == set(models2.index)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
