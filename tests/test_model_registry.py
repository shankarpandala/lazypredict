"""Tests for model registry module."""

import unittest

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TestModelRegistry(unittest.TestCase):
    """Test model registry functionality."""

    def test_classification_registry(self):
        """Test classification model registry."""
        try:
            from lazypredict.models.model_registry import (
                CLASSIFIERS,
                get_classification_models,
            )

            # Test getting all models
            models = get_classification_models()
            self.assertTrue(len(models) > 0)
            self.assertTrue(all(issubclass(model, BaseEstimator) for model in models))

            # Test getting specific models
            models = get_classification_models(["RandomForestClassifier"])
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0], RandomForestClassifier)

            # Test invalid model name
            models = get_classification_models(["InvalidModel"])
            self.assertEqual(len(models), 0)

        except ImportError as e:
            self.skipTest(f"Could not import CLASSIFIERS from classification module: {str(e)}")

    def test_regression_registry(self):
        """Test regression model registry."""
        try:
            from lazypredict.models.model_registry import (
                REGRESSORS,
                get_regression_models,
            )

            # Test getting all models
            models = get_regression_models()
            self.assertTrue(len(models) > 0)
            self.assertTrue(all(issubclass(model, BaseEstimator) for model in models))

            # Test getting specific models
            models = get_regression_models(["RandomForestRegressor"])
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0], RandomForestRegressor)

            # Test invalid model name
            models = get_regression_models(["InvalidModel"])
            self.assertEqual(len(models), 0)

        except ImportError as e:
            self.skipTest(f"Could not import REGRESSORS from regression module: {str(e)}")

    def test_classifier_filter(self):
        """Test classifier filtering."""
        try:
            from lazypredict.models.model_registry import (
                CLASSIFIERS,
                filter_models,
            )

            # Test filtering with exclude list
            filtered = filter_models(CLASSIFIERS, exclude=["RandomForestClassifier"])
            self.assertNotIn("RandomForestClassifier", filtered)

            # Test filtering with empty exclude list
            filtered = filter_models(CLASSIFIERS, exclude=[])
            self.assertEqual(filtered, CLASSIFIERS)

            # Test filtering with None exclude list
            filtered = filter_models(CLASSIFIERS)
            self.assertEqual(filtered, CLASSIFIERS)

        except ImportError:
            self.skipTest("Could not import model registry")

    def test_regressor_filter(self):
        """Test regressor filtering."""
        try:
            from lazypredict.models.model_registry import (
                REGRESSORS,
                filter_models,
            )

            # Test filtering with exclude list
            filtered = filter_models(REGRESSORS, exclude=["RandomForestRegressor"])
            self.assertNotIn("RandomForestRegressor", filtered)

            # Test filtering with empty exclude list
            filtered = filter_models(REGRESSORS, exclude=[])
            self.assertEqual(filtered, REGRESSORS)

            # Test filtering with None exclude list
            filtered = filter_models(REGRESSORS)
            self.assertEqual(filtered, REGRESSORS)

        except ImportError:
            self.skipTest("Could not import model registry")

    def test_classifier_initialization(self):
        """Test classifier initialization."""
        try:
            from lazypredict.models.model_registry import (
                get_classification_models,
            )

            models = get_classification_models()
            for model_class in models:
                # Test instantiation
                model = model_class()
                self.assertIsInstance(model, BaseEstimator)

                # Test random state if applicable
                if hasattr(model_class, "random_state"):
                    model = model_class(random_state=42)
                    self.assertEqual(model.random_state, 42)

        except ImportError:
            self.skipTest("Could not import CLASSIFIERS from classification module")

    def test_regressor_initialization(self):
        """Test regressor initialization."""
        try:
            from lazypredict.models.model_registry import get_regression_models

            models = get_regression_models()
            for model_class in models:
                # Test instantiation
                model = model_class()
                self.assertIsInstance(model, BaseEstimator)

                # Test random state if applicable
                if hasattr(model_class, "random_state"):
                    model = model_class(random_state=42)
                    self.assertEqual(model.random_state, 42)

        except ImportError:
            self.skipTest("Could not import REGRESSORS from regression module")


if __name__ == "__main__":
    unittest.main()
