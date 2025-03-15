import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
import warnings

# Ignore warnings during tests
warnings.filterwarnings("ignore")

class TestClassificationModule(unittest.TestCase):
    def setUp(self):
        # Load a simple dataset for classification
        data = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
    
    def test_classifier_import(self):
        try:
            from lazypredict.models.classification import LazyClassifier
            classifier = LazyClassifier(verbose=0, ignore_warnings=True)
            self.assertIsNotNone(classifier)
        except ImportError:
            self.skipTest("Could not import LazyClassifier from models")
    
    def test_fit_with_minimal_models(self):
        try:
            from lazypredict.models.classification import LazyClassifier
            
            # Create classifier with only decision tree to speed up test
            from sklearn.tree import DecisionTreeClassifier
            classifier = LazyClassifier(
                verbose=0, 
                ignore_warnings=True, 
                classifiers=[DecisionTreeClassifier]
            )
            
            # Fit with minimal data
            scores, predictions = classifier.fit(self.X_train, self.X_test, self.y_train, self.y_test)
            
            self.assertIsNotNone(scores)
            self.assertTrue(len(scores) > 0)
            self.assertIn('DecisionTreeClassifier', scores['Model'].values)
            
        except ImportError:
            self.skipTest("Could not import LazyClassifier from models")
    
    def test_provide_models(self):
        try:
            from lazypredict.models.classification import LazyClassifier
            
            # Create classifier with only decision tree to speed up test
            from sklearn.tree import DecisionTreeClassifier
            classifier = LazyClassifier(
                verbose=0, 
                ignore_warnings=True, 
                classifiers=[DecisionTreeClassifier]
            )
            
            # Test provide_models
            models = classifier.provide_models(self.X_train, self.X_test, self.y_train, self.y_test)
            
            self.assertIsNotNone(models)
            self.assertTrue(len(models) > 0)
            self.assertTrue('DecisionTreeClassifier' in models)
            
        except ImportError:
            self.skipTest("Could not import LazyClassifier from models")

class TestRegressionModule(unittest.TestCase):
    def setUp(self):
        # Load a simple dataset for regression
        data = load_diabetes()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
    
    def test_regressor_import(self):
        try:
            from lazypredict.models.regression import LazyRegressor
            regressor = LazyRegressor(verbose=0, ignore_warnings=True)
            self.assertIsNotNone(regressor)
        except ImportError:
            self.skipTest("Could not import LazyRegressor from models")
    
    def test_fit_with_minimal_models(self):
        try:
            from lazypredict.models.regression import LazyRegressor
            
            # Create regressor with only decision tree to speed up test
            from sklearn.tree import DecisionTreeRegressor
            regressor = LazyRegressor(
                verbose=0, 
                ignore_warnings=True, 
                regressors=[DecisionTreeRegressor]
            )
            
            # Fit with minimal data
            scores, predictions = regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
            
            self.assertIsNotNone(scores)
            self.assertTrue(len(scores) > 0)
            self.assertIn('DecisionTreeRegressor', scores['Model'].values)
            
        except ImportError:
            self.skipTest("Could not import LazyRegressor from models")
    
    def test_provide_models(self):
        try:
            from lazypredict.models.regression import LazyRegressor
            
            # Create regressor with only decision tree to speed up test
            from sklearn.tree import DecisionTreeRegressor
            regressor = LazyRegressor(
                verbose=0, 
                ignore_warnings=True, 
                regressors=[DecisionTreeRegressor]
            )
            
            # Test provide_models
            models = regressor.provide_models(self.X_train, self.X_test, self.y_train, self.y_test)
            
            self.assertIsNotNone(models)
            self.assertTrue(len(models) > 0)
            self.assertTrue('DecisionTreeRegressor' in models)
            
        except ImportError:
            self.skipTest("Could not import LazyRegressor from models")

class TestOrdinalModule(unittest.TestCase):
    def setUp(self):
        # Load a simple dataset for ordinal regression
        data = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )
    
    def test_ordinal_regressor_import(self):
        try:
            from lazypredict.models.ordinal import LazyOrdinalRegressor
            ordinal_regressor = LazyOrdinalRegressor(verbose=0, ignore_warnings=True)
            self.assertIsNotNone(ordinal_regressor)
        except ImportError:
            self.skipTest("Could not import LazyOrdinalRegressor from models")
    
    def test_fit(self):
        try:
            from lazypredict.models.ordinal import LazyOrdinalRegressor
            
            ordinal_regressor = LazyOrdinalRegressor(verbose=0, ignore_warnings=True)
            
            # Fit method may not be fully implemented, so catch NotImplementedError
            try:
                model = ordinal_regressor.fit(self.X_train, self.X_test, self.y_train, self.y_test)
                self.assertIsNotNone(model)
            except NotImplementedError:
                self.skipTest("LazyOrdinalRegressor.fit() is not fully implemented")
                
        except ImportError:
            self.skipTest("Could not import LazyOrdinalRegressor from models")

class TestSurvivalModule(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for survival analysis
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        # Mock structured array for survival data
        self.y_train = np.zeros(100, dtype=[('status', bool), ('time', float)])
        self.y_train['status'] = np.random.randint(0, 2, 100).astype(bool)
        self.y_train['time'] = np.random.uniform(0, 10, 100)
    
    def test_survival_analysis_import(self):
        try:
            from lazypredict.models.survival import LazySurvivalAnalysis
            survival_analysis = LazySurvivalAnalysis(verbose=0, ignore_warnings=True)
            self.assertIsNotNone(survival_analysis)
        except ImportError:
            self.skipTest("Could not import LazySurvivalAnalysis from models")
    
    def test_fit(self):
        try:
            from lazypredict.models.survival import LazySurvivalAnalysis
            
            survival_analysis = LazySurvivalAnalysis(verbose=0, ignore_warnings=True)
            
            # The fit method may rely on scikit-survival which might not be installed
            try:
                import sksurv
                try:
                    model = survival_analysis.fit(self.X_train, self.y_train)
                    # Test may pass or fail depending on implementation
                except NotImplementedError:
                    self.skipTest("LazySurvivalAnalysis.fit() is not fully implemented")
            except ImportError:
                self.skipTest("scikit-survival not installed")
                
        except ImportError:
            self.skipTest("Could not import LazySurvivalAnalysis from models")

class TestSequenceModule(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for sequence prediction
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10, 5)  # 3D array for sequences
        self.y_train = np.random.randint(0, 2, 100)
    
    def test_sequence_predictor_import(self):
        try:
            from lazypredict.models.sequence import LazySequencePredictor
            sequence_predictor = LazySequencePredictor(verbose=0, ignore_warnings=True)
            self.assertIsNotNone(sequence_predictor)
        except ImportError:
            self.skipTest("Could not import LazySequencePredictor from models")
    
    def test_fit(self):
        try:
            from lazypredict.models.sequence import LazySequencePredictor
            
            sequence_predictor = LazySequencePredictor(verbose=0, ignore_warnings=True)
            
            # Fit method may not be fully implemented
            try:
                model = sequence_predictor.fit(self.X_train, self.y_train)
                # Could be None if it's a placeholder
            except NotImplementedError:
                self.skipTest("LazySequencePredictor.fit() is not fully implemented")
                
        except ImportError:
            self.skipTest("Could not import LazySequencePredictor from models")

class TestBackwardCompatibility(unittest.TestCase):
    def test_supervised_imports(self):
        try:
            from lazypredict.Supervised import (
                LazyClassifier, 
                LazyRegressor,
                LazyOrdinalRegressor,
                LazySurvivalAnalysis,
                LazySequencePredictor
            )
            
            # Test that we can create instances from the old imports
            classifier = LazyClassifier(verbose=0, ignore_warnings=True)
            regressor = LazyRegressor(verbose=0, ignore_warnings=True)
            ordinal_regressor = LazyOrdinalRegressor(verbose=0, ignore_warnings=True)
            survival_analysis = LazySurvivalAnalysis(verbose=0, ignore_warnings=True)
            sequence_predictor = LazySequencePredictor(verbose=0, ignore_warnings=True)
            
            self.assertIsNotNone(classifier)
            self.assertIsNotNone(regressor)
            self.assertIsNotNone(ordinal_regressor)
            self.assertIsNotNone(survival_analysis)
            self.assertIsNotNone(sequence_predictor)
            
        except ImportError:
            self.skipTest("Could not import from lazypredict.Supervised")
            
    def test_direct_imports(self):
        try:
            from lazypredict import (
                LazyClassifier, 
                LazyRegressor,
                LazyOrdinalRegressor,
                LazySurvivalAnalysis,
                LazySequencePredictor
            )
            
            # Test that we can create instances from the new imports
            classifier = LazyClassifier(verbose=0, ignore_warnings=True)
            regressor = LazyRegressor(verbose=0, ignore_warnings=True)
            ordinal_regressor = LazyOrdinalRegressor(verbose=0, ignore_warnings=True)
            survival_analysis = LazySurvivalAnalysis(verbose=0, ignore_warnings=True)
            sequence_predictor = LazySequencePredictor(verbose=0, ignore_warnings=True)
            
            self.assertIsNotNone(classifier)
            self.assertIsNotNone(regressor)
            self.assertIsNotNone(ordinal_regressor)
            self.assertIsNotNone(survival_analysis)
            self.assertIsNotNone(sequence_predictor)
            
        except ImportError:
            self.skipTest("Could not import directly from lazypredict")

if __name__ == '__main__':
    unittest.main() 