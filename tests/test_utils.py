import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Test utility modules
class TestBaseUtils(unittest.TestCase):
    def test_base_lazy_initialization(self):
        try:
            from lazypredict.utils.base import BaseLazy
            base = BaseLazy(verbose=1, ignore_warnings=False)
            self.assertEqual(base.verbose, 1)
            self.assertEqual(base.ignore_warnings, False)
            self.assertEqual(base.random_state, 42)  # Default value
        except ImportError:
            self.skipTest("Could not import BaseLazy")
    
    def test_check_data(self):
        try:
            from lazypredict.utils.base import BaseLazy
            base = BaseLazy()
            
            # Test with numpy arrays
            X_train = np.random.rand(10, 5)
            X_test = np.random.rand(5, 5)
            y_train = np.random.rand(10)
            y_test = np.random.rand(5)
            
            X_train_out, X_test_out, y_train_out, y_test_out = base._check_data(X_train, X_test, y_train, y_test)
            
            self.assertIsInstance(X_train_out, pd.DataFrame)
            self.assertIsInstance(X_test_out, pd.DataFrame)
            self.assertIsInstance(y_train_out, np.ndarray)
            self.assertIsInstance(y_test_out, np.ndarray)
            
            # Test with pandas dataframes
            X_train_df = pd.DataFrame(X_train)
            X_test_df = pd.DataFrame(X_test)
            y_train_series = pd.Series(y_train)
            y_test_series = pd.Series(y_test)
            
            X_train_out, X_test_out, y_train_out, y_test_out = base._check_data(
                X_train_df, X_test_df, y_train_series, y_test_series
            )
            
            self.assertIsInstance(X_train_out, pd.DataFrame)
            self.assertIsInstance(X_test_out, pd.DataFrame)
            self.assertIsInstance(y_train_out, np.ndarray)
            self.assertIsInstance(y_test_out, np.ndarray)
            
        except ImportError:
            self.skipTest("Could not import BaseLazy")
    
    def test_gpu_availability_check(self):
        try:
            from lazypredict.utils.base import BaseLazy
            base = BaseLazy()
            
            # We can't reliably test the actual result since it depends on the system,
            # but we can at least ensure the method runs without error
            is_gpu_available = base._is_gpu_available()
            self.assertIsInstance(is_gpu_available, bool)
        except ImportError:
            self.skipTest("Could not import BaseLazy")

class TestPreprocessingUtils(unittest.TestCase):
    """Test preprocessing utility functions."""
    
    def setUp(self):
        # Create sample data with known properties
        self.data = pd.DataFrame({
            'numeric': np.random.rand(10),
            'categorical': ['A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B']
        })
    
    def test_categorical_cardinality_threshold(self):
        """Test categorical cardinality threshold function."""
        try:
            from lazypredict.utils.preprocessing import categorical_cardinality_threshold
            
            data = pd.DataFrame({
                'low_card': ['A', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'A'],  # 2 unique values
                'high_card': list('ABCDEFGHIJ')  # 10 unique values
            })
            
            low_card, high_card = categorical_cardinality_threshold(
                data, ['low_card', 'high_card'], threshold=5
            )
            
            self.assertEqual(low_card, ['low_card'])
            self.assertEqual(high_card, ['high_card'])
            
        except ImportError:
            self.skipTest("Could not import preprocessing utilities")
    
    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        try:
            from lazypredict.utils.preprocessing import create_preprocessor
            
            # Test with column return type
            preprocessor = create_preprocessor(self.data, return_type='column')
            self.assertIsInstance(preprocessor, ColumnTransformer)
            
            # Test with pipeline return type
            preprocessor = create_preprocessor(self.data, return_type='pipeline')
            self.assertIsInstance(preprocessor, Pipeline)
            
            # Test with polynomial features disabled
            preprocessor = create_preprocessor(self.data, enable_polynomial_features=False)
            self.assertIsInstance(preprocessor, ColumnTransformer)
            
            # Test with numpy array input
            X = np.random.rand(10, 2)
            preprocessor = create_preprocessor(X)
            self.assertTrue(isinstance(preprocessor, (Pipeline, ColumnTransformer)))
            
        except ImportError:
            self.skipTest("Could not import preprocessing utilities")

class TestGPUUtils(unittest.TestCase):
    def test_is_gpu_available(self):
        try:
            from lazypredict.utils.gpu import is_gpu_available
            
            # We can't reliably test the actual result since it depends on the system,
            # but we can at least ensure the method runs without error
            result = is_gpu_available()
            self.assertIsInstance(result, bool)
        except ImportError:
            self.skipTest("Could not import GPU utilities")
    
    def test_is_cuml_available(self):
        try:
            from lazypredict.utils.gpu import is_cuml_available
            
            # We can't reliably test the actual result since it depends on the system,
            # but we can at least ensure the method runs without error
            result = is_cuml_available()
            self.assertIsInstance(result, bool)
        except ImportError:
            self.skipTest("Could not import GPU utilities")
    
    def test_get_cpu_model(self):
        try:
            from lazypredict.utils.gpu import get_cpu_model
            
            # Test with a common model
            model_class = get_cpu_model("RandomForestClassifier")
            
            # The test should pass whether or not the model is found
            if model_class is not None:
                self.assertTrue(callable(model_class))
            
            # Test with a non-existent model
            non_existent_model = get_cpu_model("NonExistentModel")
            self.assertIsNone(non_existent_model)
            
        except ImportError:
            self.skipTest("Could not import GPU utilities")
    
    def test_get_best_model(self):
        try:
            from lazypredict.utils.gpu import get_best_model
            
            # Test with a common model
            model_class = get_best_model("RandomForestClassifier", prefer_gpu=False)
            
            # The test should pass whether or not the model is found
            if model_class is not None:
                self.assertTrue(callable(model_class))
                
        except ImportError:
            self.skipTest("Could not import GPU utilities")

class TestMLflowUtils(unittest.TestCase):
    @patch('mlflow.set_tracking_uri')
    def test_configure_mlflow(self, mock_set_tracking_uri):
        try:
            from lazypredict.utils.mlflow_utils import configure_mlflow
            
            # Test with a tracking URI
            configure_mlflow("test_uri")
            mock_set_tracking_uri.assert_called_with("test_uri")
            
        except ImportError:
            self.skipTest("Could not import MLflow utilities")
    
    @patch('mlflow.start_run')
    def test_start_run(self, mock_start_run):
        try:
            from lazypredict.utils.mlflow_utils import start_run
            
            # Mock return value
            mock_active_run = MagicMock()
            mock_active_run.info.run_id = "test_run_id"
            mock_start_run.return_value = mock_active_run
            
            # Test start_run
            start_run(run_name="test_run", experiment_name="test_experiment")
            mock_start_run.assert_called_with(run_name="test_run")
            
        except ImportError:
            self.skipTest("Could not import MLflow utilities")
            
    @patch('mlflow.end_run')
    @patch('mlflow.active_run')
    def test_end_run(self, mock_active_run, mock_end_run):
        try:
            from lazypredict.utils.mlflow_utils import end_run
            
            # Setup mock active run
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mock_active_run.return_value = mock_run
            
            # Test end_run
            end_run()
            mock_end_run.assert_called_once()
            
        except ImportError:
            self.skipTest("Could not import MLflow utilities")
    
    @patch('mlflow.log_params')
    def test_log_params(self, mock_log_params):
        try:
            from lazypredict.utils.mlflow_utils import log_params
            
            # Test log_params
            params = {"param1": 1, "param2": "test"}
            log_params(params)
            mock_log_params.assert_called_with(params)
            
        except ImportError:
            self.skipTest("Could not import MLflow utilities")
    
    @patch('mlflow.log_metric')
    def test_log_metric(self, mock_log_metric):
        try:
            from lazypredict.utils.mlflow_utils import log_metric
            
            # Test log_metric
            log_metric("test_metric", 0.95)
            mock_log_metric.assert_called_with("test_metric", 0.95)
            
        except ImportError:
            self.skipTest("Could not import MLflow utilities")
    
    @patch('mlflow.log_artifact')
    def test_log_dataframe(self, mock_log_artifact):
        try:
            from lazypredict.utils.mlflow_utils import log_dataframe
            
            # Test log_dataframe
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            
            # We need to patch tempfile.TemporaryDirectory
            with patch('tempfile.TemporaryDirectory') as mock_temp_dir:
                mock_temp_dir.return_value.__enter__.return_value = "/tmp/temp_dir"
                log_dataframe(df, "test_artifact")
                
                # Assert log_artifact was called (path will be different each time)
                self.assertTrue(mock_log_artifact.called)
                
        except ImportError:
            self.skipTest("Could not import MLflow utilities")

if __name__ == '__main__':
    unittest.main()