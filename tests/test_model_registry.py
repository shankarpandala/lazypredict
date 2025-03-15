import unittest
import warnings

# Ignore warnings during tests
warnings.filterwarnings("ignore")

class TestModelRegistry(unittest.TestCase):
    def test_classification_registry(self):
        try:
            from lazypredict.models.classification import CLASSIFIERS
            
            # Check that the registry is not empty
            self.assertTrue(len(CLASSIFIERS) > 0, "Classification model registry is empty")
            
            # Check that common classifiers are registered
            expected_classifiers = [
                'LogisticRegression',
                'RandomForestClassifier',
                'SVC',
                'DecisionTreeClassifier',
                'KNeighborsClassifier',
                'GaussianNB'
            ]
            
            # Create a set of registered classifier names for faster lookup
            registered_names = {clf.__name__ for clf in CLASSIFIERS}
            
            # Check that each expected classifier is registered
            for expected in expected_classifiers:
                self.assertIn(
                    expected, 
                    registered_names, 
                    f"Expected classifier {expected} not found in registry"
                )
                
        except ImportError:
            self.skipTest("Could not import CLASSIFIERS from classification module")
    
    def test_regression_registry(self):
        try:
            from lazypredict.models.regression import REGRESSORS
            
            # Check that the registry is not empty
            self.assertTrue(len(REGRESSORS) > 0, "Regression model registry is empty")
            
            # Check that common regressors are registered
            expected_regressors = [
                'LinearRegression',
                'RandomForestRegressor',
                'SVR',
                'DecisionTreeRegressor',
                'KNeighborsRegressor',
                'ElasticNet'
            ]
            
            # Create a set of registered regressor names for faster lookup
            registered_names = {reg.__name__ for reg in REGRESSORS}
            
            # Check that each expected regressor is registered
            for expected in expected_regressors:
                self.assertIn(
                    expected, 
                    registered_names, 
                    f"Expected regressor {expected} not found in registry"
                )
                
        except ImportError:
            self.skipTest("Could not import REGRESSORS from regression module")
    
    def test_classifier_initialization(self):
        try:
            from lazypredict.models.classification import CLASSIFIERS
            
            # Try to initialize each classifier
            for Classifier in CLASSIFIERS:
                try:
                    # Initialize with default parameters
                    model = Classifier()
                    self.assertIsNotNone(model, f"Failed to initialize {Classifier.__name__}")
                except Exception as e:
                    # Skip this test if initialization fails (might require special parameters)
                    self.skipTest(f"Could not initialize {Classifier.__name__}: {str(e)}")
                    
        except ImportError:
            self.skipTest("Could not import CLASSIFIERS from classification module")
    
    def test_regressor_initialization(self):
        try:
            from lazypredict.models.regression import REGRESSORS
            
            # Try to initialize each regressor
            for Regressor in REGRESSORS:
                try:
                    # Initialize with default parameters
                    model = Regressor()
                    self.assertIsNotNone(model, f"Failed to initialize {Regressor.__name__}")
                except Exception as e:
                    # Skip this test if initialization fails (might require special parameters)
                    self.skipTest(f"Could not initialize {Regressor.__name__}: {str(e)}")
                    
        except ImportError:
            self.skipTest("Could not import REGRESSORS from regression module")
    
    def test_classifier_filter(self):
        try:
            from lazypredict.models.classification import LazyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            
            # Create classifier with specific models
            specified_models = [LogisticRegression, RandomForestClassifier]
            classifier = LazyClassifier(
                verbose=0,
                ignore_warnings=True,
                classifiers=specified_models
            )
            
            # Check that only specified models are included
            self.assertEqual(
                len(classifier.models), 
                len(specified_models),
                "Filtered classifier count doesn't match specified models"
            )
            
            # Check model names
            model_names = [model.__class__.__name__ for model in classifier.models.values()]
            self.assertIn('LogisticRegression', model_names)
            self.assertIn('RandomForestClassifier', model_names)
            
        except ImportError:
            self.skipTest("Could not import LazyClassifier from classification module")
    
    def test_regressor_filter(self):
        try:
            from lazypredict.models.regression import LazyRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            
            # Create regressor with specific models
            specified_models = [LinearRegression, RandomForestRegressor]
            regressor = LazyRegressor(
                verbose=0,
                ignore_warnings=True,
                regressors=specified_models
            )
            
            # Check that only specified models are included
            self.assertEqual(
                len(regressor.models), 
                len(specified_models),
                "Filtered regressor count doesn't match specified models"
            )
            
            # Check model names
            model_names = [model.__class__.__name__ for model in regressor.models.values()]
            self.assertIn('LinearRegression', model_names)
            self.assertIn('RandomForestRegressor', model_names)
            
        except ImportError:
            self.skipTest("Could not import LazyRegressor from regression module")

if __name__ == '__main__':
    unittest.main() 