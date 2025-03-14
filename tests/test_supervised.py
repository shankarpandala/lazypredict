import unittest
import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier, LazyRegressor, LazyOrdinalRegressor, LazySurvivalAnalysis, LazySequencePredictor
from sklearn.datasets import load_breast_cancer, load_boston, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import mlflow
import torch
try:
    from cuml.ensemble import RandomForestClassifier as cumlRF
    gpu_available = torch.cuda.is_available()
except ImportError:
    gpu_available = False

# Use cuML's RandomForest if GPU is available, otherwise use scikit-learn's
if gpu_available:
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
        best_params = self.classifier.fit_optimize(self.X_train, self.y_train, RandomForestClassifier)
        self.assertIsNotNone(best_params)
        self.assertTrue('n_estimators' in best_params)

    def test_mlflow_integration(self):
        global GLOBAL_MLFLOW_TRACKING_URI
        GLOBAL_MLFLOW_TRACKING_URI = "http://localhost:5000"
        self.classifier.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        # Check if MLflow run was started
        self.assertTrue(mlflow.active_run() is not None)

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
        best_params = self.regressor.fit_optimize(self.X_train, self.y_train, RandomForestClassifier)
        self.assertIsNotNone(best_params)
        self.assertTrue('n_estimators' in best_params)

    def test_mlflow_integration(self):
        global GLOBAL_MLFLOW_TRACKING_URI
        GLOBAL_MLFLOW_TRACKING_URI = "http://localhost:5000"
        self.regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        # Check if MLflow run was started
        self.assertTrue(mlflow.active_run() is not None)

class TestLazyOrdinalRegressor(unittest.TestCase):
    def setUp(self):
        data = fetch_openml(name='credit-g', version=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        self.ordinal_regressor = LazyOrdinalRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.ordinal_regressor, LazyOrdinalRegressor)

    def test_fit(self):
        model = self.ordinal_regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsNotNone(model)

class TestLazySurvivalAnalysis(unittest.TestCase):
    def setUp(self):
        X, y = load_whas500()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.survival_analysis = LazySurvivalAnalysis(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.survival_analysis, LazySurvivalAnalysis)

    def test_fit(self):
        model = self.survival_analysis.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model)

class TestLazySequencePredictor(unittest.TestCase):
    def setUp(self):
        # Placeholder setup for sequence prediction
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            np.random.rand(100, 10), np.random.randint(0, 2, 100), test_size=0.2, random_state=42
        )
        self.sequence_predictor = LazySequencePredictor(verbose=0, ignore_warnings=True, custom_metric=None)

    def test_initialization(self):
        self.assertIsInstance(self.sequence_predictor, LazySequencePredictor)

    def test_fit(self):
        model = self.sequence_predictor.fit(self.X_train, self.y_train)
        self.assertIsNone(model)  # Placeholder test, as fit returns None

if __name__ == '__main__':
    unittest.main()