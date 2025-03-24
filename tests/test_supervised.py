import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from lazypredict import LazyClassifier, LazyRegressor, Supervised

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
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                data.data, data.target, test_size=0.2, random_state=42
            )
        )
        self.classifier = LazyClassifier(
            verbose=0, ignore_warnings=True, custom_metric=None
        )

    def test_initialization(self):
        self.assertIsInstance(self.classifier, LazyClassifier)

    def test_fit(self):
        scores, predictions = self.classifier.fit(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertIsNotNone(scores)
        self.assertTrue(len(scores) > 0)

    def test_provide_models(self):
        self.classifier.fit(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        models = self.classifier.provide_models(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertIsNotNone(models)
        self.assertTrue(len(models) > 0)

    def test_fit_optimize(self):
        try:
            import optuna

            best_params = self.classifier.fit_optimize(
                self.X_train, self.y_train, RandomForestClassifier
            )
            self.assertIsNotNone(best_params)
            self.assertTrue("n_estimators" in best_params)
        except ImportError:
            self.skipTest("Optuna not installed. Skipping test_fit_optimize")

    @unittest.skipIf(not MLFLOW_AVAILABLE, "MLflow not installed")
    def test_mlflow_integration(self):
        # Mock MLflow tracking URI
        mlflow_tracking_uri = "file:///tmp/mlflow-test"
        try:
            self.classifier.fit(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                mlflow_tracking_uri=mlflow_tracking_uri,
            )
            # Check if MLflow run was started
            self.assertIsNotNone(mlflow.active_run())
            mlflow.end_run()  # Clean up the run
        except Exception as e:
            self.fail(f"MLflow integration failed: {e}")


class TestLazyRegressor(unittest.TestCase):
    def setUp(self):
        data = load_diabetes()
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                data.data, data.target, test_size=0.2, random_state=42
            )
        )
        self.regressor = LazyRegressor(
            verbose=0, ignore_warnings=True, custom_metric=None
        )

    def test_initialization(self):
        self.assertIsInstance(self.regressor, LazyRegressor)

    def test_fit(self):
        scores, predictions = self.regressor.fit(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertIsNotNone(scores)
        self.assertTrue(len(scores) > 0)

    def test_provide_models(self):
        self.regressor.fit(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        models = self.regressor.provide_models(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.assertIsNotNone(models)
        self.assertTrue(len(models) > 0)

    def test_fit_optimize(self):
        try:
            import optuna

            best_params = self.regressor.fit_optimize(
                self.X_train, self.y_train, RandomForestClassifier
            )
            self.assertIsNotNone(best_params)
            self.assertTrue("n_estimators" in best_params)
        except ImportError:
            self.skipTest("Optuna not installed. Skipping test_fit_optimize")

    @unittest.skipIf(not MLFLOW_AVAILABLE, "MLflow not installed")
    def test_mlflow_integration(self):
        # Mock MLflow tracking URI
        mlflow_tracking_uri = "file:///tmp/mlflow-test"
        try:
            self.regressor.fit(
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
                mlflow_tracking_uri=mlflow_tracking_uri,
            )
            # Check if MLflow run was started
            self.assertIsNotNone(mlflow.active_run())
            mlflow.end_run()  # Clean up the run
        except Exception as e:
            self.fail(f"MLflow integration failed: {e}")


"""Tests for supervised learning module."""

import unittest

from lazypredict import (
    LazyClassifier,
    LazyRegressor,
    Supervised,
)


class TestSupervisedModule(unittest.TestCase):
    """Test supervised learning module functionality."""

    def test_supervised_alias(self):
        """Test Supervised alias points to LazyClassifier."""
        self.assertEqual(Supervised, LazyClassifier)


if __name__ == "__main__":
    unittest.main()
