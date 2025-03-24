"""Tests for model initialization and fitting."""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


class TestClassificationModule(unittest.TestCase):
    """Test classification module functionality."""

    def setUp(self):
        """Set up test data."""
        # Create classification dataset
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42,
        )
        self.X_train = pd.DataFrame(
            X[:80], columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        self.X_test = pd.DataFrame(
            X[80:], columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        self.y_train = y[:80]
        self.y_test = y[80:]

    def test_classifier_import(self):
        """Test classifier import."""
        try:
            from lazypredict import LazyClassifier

            classifier = LazyClassifier()
            self.assertIsNotNone(classifier)
        except ImportError:
            self.skipTest("Could not import LazyClassifier")

    def test_fit_with_minimal_models(self):
        """Test classifier fitting with minimal set of models."""
        try:
            from lazypredict import LazyClassifier

            # Initialize with only decision tree
            classifier = LazyClassifier(classifiers=["DecisionTreeClassifier"])

            # Fit and get scores
            scores, predictions = classifier.fit(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertTrue(len(scores) > 0)
            self.assertTrue("Model" in scores.columns)
            self.assertTrue("Accuracy" in scores.columns)

        except ImportError:
            self.skipTest("Could not import LazyClassifier")

    def test_provide_models(self):
        """Test providing specific models."""
        try:
            from lazypredict import LazyClassifier

            # Initialize with specific models
            models = ["RandomForestClassifier", "DecisionTreeClassifier"]
            classifier = LazyClassifier(classifiers=models)

            # Check models are initialized
            self.assertTrue(len(classifier.models) > 0)

            # Check we can fit with these models
            scores, predictions = classifier.fit(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertEqual(len(scores), len(models))

        except ImportError:
            self.skipTest("Could not import LazyClassifier")


class TestRegressionModule(unittest.TestCase):
    """Test regression module functionality."""

    def setUp(self):
        """Set up test data."""
        # Create regression dataset
        X, y = make_regression(
            n_samples=100, n_features=5, n_informative=3, random_state=42
        )
        self.X_train = pd.DataFrame(
            X[:80], columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        self.X_test = pd.DataFrame(
            X[80:], columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        self.y_train = y[:80]
        self.y_test = y[80:]

    def test_regressor_import(self):
        """Test regressor import."""
        try:
            from lazypredict import LazyRegressor

            regressor = LazyRegressor()
            self.assertIsNotNone(regressor)
        except ImportError:
            self.skipTest("Could not import LazyRegressor")

    def test_fit_with_minimal_models(self):
        """Test regressor fitting with minimal set of models."""
        try:
            from lazypredict import LazyRegressor

            # Initialize with only decision tree
            regressor = LazyRegressor(regressors=["DecisionTreeRegressor"])

            # Fit and get scores
            scores, predictions = regressor.fit(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertTrue(len(scores) > 0)
            self.assertTrue("Model" in scores.columns)
            self.assertTrue("R-Squared" in scores.columns)

        except ImportError:
            self.skipTest("Could not import LazyRegressor")

    def test_provide_models(self):
        """Test providing specific models."""
        try:
            from lazypredict import LazyRegressor

            # Initialize with specific models
            models = ["RandomForestRegressor", "DecisionTreeRegressor"]
            regressor = LazyRegressor(regressors=models)

            # Check models are initialized
            self.assertTrue(len(regressor.models) > 0)

            # Check we can fit with these models
            scores, predictions = regressor.fit(
                self.X_train, self.X_test, self.y_train, self.y_test
            )

            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertEqual(len(scores), len(models))

        except ImportError:
            self.skipTest("Could not import LazyRegressor")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility."""

    def test_supervised_imports(self):
        """Test imports from Supervised module."""
        try:
            from lazypredict.Supervised import Supervised

            model = Supervised()
            self.assertIsNotNone(model)
        except ImportError:
            self.skipTest("Could not import from lazypredict.Supervised")

    def test_direct_imports(self):
        """Test direct imports from lazypredict."""
        try:
            from lazypredict import LazyClassifier, LazyRegressor, Supervised

            self.assertTrue(callable(LazyClassifier))
            self.assertTrue(callable(LazyRegressor))
            self.assertTrue(callable(Supervised))
        except ImportError:
            self.skipTest("Could not import directly from lazypredict")


if __name__ == "__main__":
    unittest.main()
