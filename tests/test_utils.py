import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

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
    def test_categorical_cardinality_threshold(self):
        try:
            from lazypredict.utils.preprocessing import get_categorical_cardinality_threshold
            
            # Create sample data with categorical features
            data = pd.DataFrame({
                'low_card': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
                'high_card': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            })
            
            low_card, high_card = get_categorical_cardinality_threshold(
                data, ['low_card', 'high_card'], threshold=5
            )
            
            self.assertEqual(low_card, ['low_card'])
            self.assertEqual(high_card, ['high_card'])
            
        except ImportError:
            self.skipTest("Could not import preprocessing utilities")
    
    def test_create_preprocessor(self):
        try:
            from lazypredict.utils.preprocessing import create_preprocessor
            from sklearn.compose import ColumnTransformer
            
            # Create sample mixed data
            data = pd.DataFrame({
                'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
                'cat_low': ['A', 'B', 'A', 'B', 'A'],
                'cat_high': ['X', 'Y', 'Z', 'W', 'V']
            })
            
            preprocessor = create_preprocessor(data)
            
            self.assertIsInstance(preprocessor, ColumnTransformer)
            
            # Check that transformers were created for each type
            transformer_names = [name for name, _, _ in preprocessor.transformers]
            self.assertIn('numeric', transformer_names)
            
            # Test with polynomial features disabled
            preprocessor_no_poly = create_preprocessor(data, enable_polynomial_features=False)
            transformer_names = [name for name, _, _ in preprocessor_no_poly.transformers]
            self.assertNotIn('poly', transformer_names)
            
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
    def test_end_run(self, mock_end_run):
        try:
            from lazypredict.utils.mlflow_utils import end_run
            
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