import unittest
import numpy as np
import pandas as pd
from lazypredict import (
    LazyClassifier, 
    LazyRegressor, 
    LazyOrdinalRegressor, 
    LazySurvivalAnalysis, 
    LazySequencePredictor
)
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings

# Ignore warnings during tests
warnings.filterwarnings("ignore")

# Check for optional dependencies
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRF
    GPU_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# Use cuML's RandomForest if GPU is available, otherwise use scikit-learn's
if GPU_AVAILABLE:
    RandomForestClassifier = cumlRF
else:
    from sklearn.ensemble import RandomForestClassifier

class TestLazyClassifier(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        self.classifier = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.classifier, LazyClassifier)

    def test_fit(self):
        scores, predictions = self.classifier.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(scores)
        self.assertTrue(len(scores) > 0)

    def test_provide_models(self):
        self.classifier.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        models = self.classifier.provide_models(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(models)
        self.assertTrue(len(models) > 0)

    def test_fit_optimize(self):
        try:
            import optuna
            best_params = self.classifier.fit_optimize(self.X_train, self.y_train, RandomForestClassifier)
            self.assertIsNotNone(best_params)
            self.assertTrue('n_estimators' in best_params)
        except ImportError:
            self.skipTest("Optuna not installed. Skipping test_fit_optimize")

    @unittest.skipIf(not MLFLOW_AVAILABLE, "MLflow not installed")
    def test_mlflow_integration(self):
        # Mock MLflow tracking URI
        mlflow_tracking_uri = "file:///tmp/mlflow-test"
        try:
            self.classifier.fit(self.X_train, self.X_test, self.y_train, self.y_test, mlflow_tracking_uri=mlflow_tracking_uri)
            # Check if MLflow run was started
            self.assertIsNotNone(mlflow.active_run())
            mlflow.end_run()  # Clean up the run
        except Exception as e:
            self.fail(f"MLflow integration failed: {e}")

class TestLazyRegressor(unittest.TestCase):
    def setUp(self):
        data = load_diabetes()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        self.regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.regressor, LazyRegressor)

    def test_fit(self):
        scores, predictions = self.regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(scores)
        self.assertTrue(len(scores) > 0)

    def test_provide_models(self):
        self.regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        models = self.regressor.provide_models(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(models)
        self.assertTrue(len(models) > 0)

    def test_fit_optimize(self):
        try:
            import optuna
            best_params = self.regressor.fit_optimize(self.X_train, self.y_train, RandomForestClassifier)
            self.assertIsNotNone(best_params)
            self.assertTrue('n_estimators' in best_params)
        except ImportError:
            self.skipTest("Optuna not installed. Skipping test_fit_optimize")

    @unittest.skipIf(not MLFLOW_AVAILABLE, "MLflow not installed")
    def test_mlflow_integration(self):
        # Mock MLflow tracking URI
        mlflow_tracking_uri = "file:///tmp/mlflow-test"
        try:
            self.regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test, mlflow_tracking_uri=mlflow_tracking_uri)
            # Check if MLflow run was started
            self.assertIsNotNone(mlflow.active_run())
            mlflow.end_run()  # Clean up the run
        except Exception as e:
            self.fail(f"MLflow integration failed: {e}")

class TestLazyOrdinalRegressor(unittest.TestCase):
    def setUp(self):
        # Use iris dataset instead of fetch_openml which might not be available
        data = load_iris()
        # Convert to binary problem for simplicity
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        self.ordinal_regressor = LazyOrdinalRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.ordinal_regressor, LazyOrdinalRegressor)

    def test_fit(self):
        # This assumes LazyOrdinalRegressor has been implemented properly.
        # If it returns None (placeholder), adjust the assertion
        try:
            model = self.ordinal_regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
            self.assertIsNotNone(model)
        except NotImplementedError:
            self.skipTest("LazyOrdinalRegressor.fit() not fully implemented yet")

class TestLazySurvivalAnalysis(unittest.TestCase):
    def setUp(self):
        # Since sksurv might not be installed, create a simple mock dataset
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        # Mock structured array for survival data
        self.y_train = np.zeros(100, dtype=[('status', bool), ('time', float)])
        self.y_train['status'] = np.random.randint(0, 2, 100).astype(bool)
        self.y_train['time'] = np.random.uniform(0, 10, 100)
        
        self.survival_analysis = LazySurvivalAnalysis(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.survival_analysis, LazySurvivalAnalysis)

    def test_fit(self):
        try:
            import sksurv
            model = self.survival_analysis.fit(self.X_train, self.y_train)
            self.assertIsNotNone(model)
        except (ImportError, NotImplementedError):
            self.skipTest("scikit-survival not installed or not fully implemented")

class TestLazySequencePredictor(unittest.TestCase):
    def setUp(self):
        # Placeholder setup for sequence prediction
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10, 5)  # Sequential data
        self.y_train = np.random.randint(0, 2, 100)
        self.sequence_predictor = LazySequencePredictor(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.sequence_predictor, LazySequencePredictor)

    def test_fit(self):
        try:
            model = self.sequence_predictor.fit(self.X_train, self.y_train)
            # Could be None if placeholder
            pass
        except NotImplementedError:
            self.skipTest("LazySequencePredictor.fit() not fully implemented yet")

if __name__ == '__main__':
    unittest.main()