import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

# Ignore warnings during tests
warnings.filterwarnings("ignore")


class TestPackageIntegration(unittest.TestCase):
    def setUp(self):
        # Load datasets
        iris = load_iris()
        diabetes = load_diabetes()

        # Classification data
        self.X_class, self.X_class_test, self.y_class, self.y_class_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        # Binary data
        self.y_binary = (iris.target > 0).astype(int)
        (
            self.X_binary,
            self.X_binary_test,
            self.y_binary,
            self.y_binary_test,
        ) = train_test_split(iris.data, self.y_binary, test_size=0.2, random_state=42)

        # Regression data
        self.X_reg, self.X_reg_test, self.y_reg, self.y_reg_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.2, random_state=42
        )

        # Convert to pandas for some tests
        feat_names = [f"feature_{i}" for i in range(iris.data.shape[1])]
        self.X_class_df = pd.DataFrame(self.X_class, columns=feat_names)
        self.X_class_test_df = pd.DataFrame(self.X_class_test, columns=feat_names)

    def test_classification_direct_import(self):
        try:
            # Test with minimal models (just one) to keep tests fast
            from sklearn.tree import DecisionTreeClassifier

            from lazypredict import LazyClassifier

            classifier = LazyClassifier(
                verbose=0,
                ignore_warnings=True,
                custom_metric=None,
                classifiers=[DecisionTreeClassifier],
            )

            # Test with numpy arrays
            scores, predictions = classifier.fit(
                self.X_class,
                self.X_class_test,
                self.y_class,
                self.y_class_test,
            )

            self.assertIsNotNone(scores)
            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertTrue("DecisionTreeClassifier" in scores["Model"].values)

            # Test with pandas DataFrame
            scores_df, predictions_df = classifier.fit(
                self.X_class_df,
                self.X_class_test_df,
                self.y_class,
                self.y_class_test,
            )

            self.assertIsNotNone(scores_df)
            self.assertTrue(isinstance(scores_df, pd.DataFrame))
            self.assertTrue("DecisionTreeClassifier" in scores_df["Model"].values)

        except ImportError:
            self.skipTest("Could not import LazyClassifier directly from lazypredict")

    def test_regression_direct_import(self):
        try:
            # Test with minimal models (just one) to keep tests fast
            from sklearn.tree import DecisionTreeRegressor

            from lazypredict import LazyRegressor

            regressor = LazyRegressor(
                verbose=0,
                ignore_warnings=True,
                custom_metric=None,
                regressors=[DecisionTreeRegressor],
            )

            # Test with numpy arrays
            scores, predictions = regressor.fit(
                self.X_reg, self.X_reg_test, self.y_reg, self.y_reg_test
            )

            self.assertIsNotNone(scores)
            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertTrue("DecisionTreeRegressor" in scores["Model"].values)

        except ImportError:
            self.skipTest("Could not import LazyRegressor directly from lazypredict")

    def test_supervised_backwards_compatibility(self):
        try:
            # Test with minimal models (just one) to keep tests fast
            from sklearn.tree import (
                DecisionTreeClassifier,
                DecisionTreeRegressor,
            )

            from lazypredict.Supervised import LazyClassifier, LazyRegressor

            # Test classifier
            classifier = LazyClassifier(
                verbose=0,
                ignore_warnings=True,
                custom_metric=None,
                classifiers=[DecisionTreeClassifier],
            )

            scores, predictions = classifier.fit(
                self.X_class,
                self.X_class_test,
                self.y_class,
                self.y_class_test,
            )

            self.assertIsNotNone(scores)
            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertTrue("DecisionTreeClassifier" in scores["Model"].values)

            # Test regressor
            regressor = LazyRegressor(
                verbose=0,
                ignore_warnings=True,
                custom_metric=None,
                regressors=[DecisionTreeRegressor],
            )

            scores, predictions = regressor.fit(
                self.X_reg, self.X_reg_test, self.y_reg, self.y_reg_test
            )

            self.assertIsNotNone(scores)
            self.assertTrue(isinstance(scores, pd.DataFrame))
            self.assertTrue("DecisionTreeRegressor" in scores["Model"].values)

        except ImportError:
            self.skipTest("Could not import from lazypredict.Supervised")

    def test_custom_metrics(self):
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.tree import DecisionTreeClassifier

            from lazypredict import LazyClassifier

            # Define a custom metric function
            def custom_accuracy(y_true, y_pred):
                return accuracy_score(y_true, y_pred)

            # Test with custom metric
            classifier = LazyClassifier(
                verbose=0,
                ignore_warnings=True,
                custom_metric=custom_accuracy,
                classifiers=[DecisionTreeClassifier],
            )

            scores, predictions = classifier.fit(
                self.X_binary,
                self.X_binary_test,
                self.y_binary,
                self.y_binary_test,
            )

            self.assertIsNotNone(scores)
            self.assertTrue("Custom Metric" in scores.columns)

        except ImportError:
            self.skipTest("Could not import LazyClassifier directly from lazypredict")

    def test_model_types_compatibility(self):
        try:
            from lazypredict import LazyClassifier as DirectClassifier
            from lazypredict import LazyRegressor as DirectRegressor
            from lazypredict.models.classification import LazyClassifier as ClassificationLazy
            from lazypredict.models.regression import LazyRegressor as RegressionLazy

            # Verify these are the same class
            self.assertEqual(ClassificationLazy, DirectClassifier)
            self.assertEqual(RegressionLazy, DirectRegressor)

        except ImportError:
            self.skipTest("Could not import LazyClassifier or LazyRegressor")


if __name__ == "__main__":
    unittest.main()
