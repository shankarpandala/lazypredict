import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings

# Ignore warnings during tests
warnings.filterwarnings("ignore")

class TestLazyBase(unittest.TestCase):
    def setUp(self):
        # Load a simple dataset
        data = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        # Convert to pandas DataFrame for some tests
        self.X_train_df = pd.DataFrame(self.X_train, columns=[f'feature_{i}' for i in range(self.X_train.shape[1])])
        self.X_test_df = pd.DataFrame(self.X_test, columns=[f'feature_{i}' for i in range(self.X_test.shape[1])])
    
    def test_base_import(self):
        try:
            from lazypredict.base import Lazy
            lazy = Lazy(verbose=0, ignore_warnings=True)
            self.assertIsNotNone(lazy)
        except ImportError:
            self.skipTest("Could not import Lazy base class")
    
    def test_base_methods(self):
        try:
            from lazypredict.base import Lazy
            
            lazy = Lazy(verbose=0, ignore_warnings=True)
            
            # Test basic methods, expecting NotImplementedError for abstract methods
            try:
                lazy.fit(self.X_train, self.X_test, self.y_train, self.y_test)
                self.fail("Expected NotImplementedError")
            except NotImplementedError:
                pass  # Expected
                
            try:
                lazy.provide_models(self.X_train, self.X_test, self.y_train, self.y_test)
                self.fail("Expected NotImplementedError")
            except NotImplementedError:
                pass  # Expected
                
        except ImportError:
            self.skipTest("Could not import Lazy base class")
    
    def test_parameter_validation(self):
        try:
            from lazypredict.base import Lazy
            
            # Test with invalid parameters
            with self.assertRaises(ValueError):
                lazy = Lazy(verbose=-1, ignore_warnings=True)
                
            # Test with valid parameters
            lazy = Lazy(verbose=0, ignore_warnings=True)
            lazy = Lazy(verbose=1, ignore_warnings=True)
            lazy = Lazy(verbose=2, ignore_warnings=True)
                
        except ImportError:
            self.skipTest("Could not import Lazy base class")

class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create test data
        np.random.seed(42)
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 2, 100)
        
        # Create pandas DataFrame
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(self.X.shape[1])])
        
    def test_utils_import(self):
        try:
            from lazypredict.utils import get_model_name, check_X_y
            self.assertTrue(callable(get_model_name))
            self.assertTrue(callable(check_X_y))
        except ImportError:
            self.skipTest("Could not import utility functions")
    
    def test_get_model_name(self):
        try:
            from lazypredict.utils import get_model_name
            from sklearn.linear_model import LogisticRegression
            
            # Test with sklearn model
            model_name = get_model_name(LogisticRegression())
            self.assertEqual(model_name, "LogisticRegression")
            
            # Test with string
            model_name = get_model_name("SomeModel")
            self.assertEqual(model_name, "SomeModel")
            
            # Test with class
            model_name = get_model_name(LogisticRegression)
            self.assertEqual(model_name, "LogisticRegression")
            
        except ImportError:
            self.skipTest("Could not import get_model_name")
    
    def test_check_X_y(self):
        try:
            from lazypredict.utils import check_X_y
            
            # Test with numpy arrays
            X_processed, y_processed = check_X_y(self.X, self.y)
            self.assertTrue(isinstance(X_processed, np.ndarray))
            self.assertTrue(isinstance(y_processed, np.ndarray))
            
            # Test with pandas DataFrame
            X_processed, y_processed = check_X_y(self.X_df, self.y)
            self.assertTrue(isinstance(X_processed, np.ndarray))
            self.assertTrue(isinstance(y_processed, np.ndarray))
            
            # Test with mixed types
            X_processed, y_processed = check_X_y(self.X, pd.Series(self.y))
            self.assertTrue(isinstance(X_processed, np.ndarray))
            self.assertTrue(isinstance(y_processed, np.ndarray))
            
        except ImportError:
            self.skipTest("Could not import check_X_y")

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Load a simple dataset
        data = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
        
        # Create binary classification data
        self.y_binary = (data.target > 0).astype(int)
        self.X_train_binary, self.X_test_binary, self.y_train_binary, self.y_test_binary = train_test_split(
            data.data, self.y_binary, test_size=0.2, random_state=42
        )
        
        # Create regression data
        np.random.seed(42)
        self.y_regression = np.random.rand(data.data.shape[0])
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
            data.data, self.y_regression, test_size=0.2, random_state=42
        )
    
    def test_metrics_import(self):
        try:
            from lazypredict.metrics import (
                get_classification_metrics,
                get_regression_metrics
            )
            
            self.assertTrue(callable(get_classification_metrics))
            self.assertTrue(callable(get_regression_metrics))
            
        except ImportError:
            self.skipTest("Could not import metrics functions")
    
    def test_classification_metrics(self):
        try:
            from lazypredict.metrics import get_classification_metrics
            from sklearn.ensemble import RandomForestClassifier
            
            # Train a simple model
            model = RandomForestClassifier(random_state=42)
            model.fit(self.X_train_binary, self.y_train_binary)
            
            # Get predictions
            y_pred = model.predict(self.X_test_binary)
            
            # Calculate metrics
            metrics = get_classification_metrics(
                self.y_test_binary,
                y_pred,
                y_score=model.predict_proba(self.X_test_binary)[:, 1] if hasattr(model, "predict_proba") else None
            )
            
            # Check metrics
            self.assertIn('Accuracy', metrics)
            self.assertIn('F1 Score', metrics)
            self.assertIn('ROC AUC', metrics)
            
        except ImportError:
            self.skipTest("Could not import classification metrics function")
    
    def test_regression_metrics(self):
        try:
            from lazypredict.metrics import get_regression_metrics
            from sklearn.ensemble import RandomForestRegressor
            
            # Train a simple model
            model = RandomForestRegressor(random_state=42)
            model.fit(self.X_train_reg, self.y_train_reg)
            
            # Get predictions
            y_pred = model.predict(self.X_test_reg)
            
            # Calculate metrics
            metrics = get_regression_metrics(self.y_test_reg, y_pred)
            
            # Check metrics
            self.assertIn('R-Squared', metrics)
            self.assertIn('MAE', metrics)
            self.assertIn('RMSE', metrics)
            
        except ImportError:
            self.skipTest("Could not import regression metrics function")

if __name__ == '__main__':
    unittest.main() 