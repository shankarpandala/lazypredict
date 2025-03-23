"""
Base module with common functionality and utilities for lazypredict.
"""
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators

# Configure logging
logger = logging.getLogger("lazypredict")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Define custom warning behavior
def custom_formatwarning(msg, *args, **kwargs):
    """Format warnings in a cleaner way."""
    return f"Warning: {msg}\n"

warnings.formatwarning = custom_formatwarning

class BaseLazy:
    """Base class for all lazy predictors."""
    
    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
    ):
        """Initialize the base lazy predictor.
        
        Parameters
        ----------
        verbose : int, optional (default=0)
            Controls the verbosity of the predictor.
            0 = no output
            1 = basic output
            2 = detailed output
        ignore_warnings : bool, optional (default=True)
            Whether to ignore warnings during model fitting.
        custom_metric : callable, optional (default=None)
            Custom evaluation metric to use.
        predictions : bool, optional (default=False)
            Whether to return predictions along with the models.
        random_state : int, optional (default=42)
            Random state for reproducibility.
        """
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.random_state = random_state
        self.models = {}
        
        # Configure logging based on verbosity
        if verbose == 0:
            logger.setLevel(logging.WARNING)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)
            
        # Configure warnings
        if ignore_warnings:
            warnings.filterwarnings("ignore")
        else:
            warnings.filterwarnings("default")
    
    def _check_data(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        """Check input data and convert to appropriate format.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        y_train : array-like of shape (n_samples,)
            Training labels.
        y_test : array-like of shape (n_samples,), optional (default=None)
            Test labels.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Optional[np.ndarray]]
            Processed input data.
        """
        # Convert arrays to dataframes if necessary
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
            
        # Convert y to numpy arrays
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
        if y_test is not None:
            if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
                y_test = y_test.values
                
        return X_train, X_test, y_train, y_test
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available.
        
        Returns
        -------
        bool
            True if GPU is available, False otherwise.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("torch not installed. GPU acceleration not available.")
            return False
            
    def _get_estimators(
        self, 
        estimator_type: str,
        excluded_estimators: List[str]
    ) -> List[Tuple[str, BaseEstimator]]:
        """Get all estimators of a specific type.
        
        Parameters
        ----------
        estimator_type : str
            Type of estimator ('classifier' or 'regressor').
        excluded_estimators : List[str]
            List of estimator names to exclude.
            
        Returns
        -------
        List[Tuple[str, BaseEstimator]]
            List of (name, estimator) tuples.
        """
        try:
            estimators = all_estimators(type_filter=estimator_type)
            return [
                est for est in estimators
                if est[0] not in excluded_estimators
            ]
        except Exception as e:
            logger.error(f"Error getting estimators: {e}")
            return []

def check_X_y(X, y):
    """Check and prepare X and y for model training."""
    # Convert pandas DataFrames to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        
    # Convert pandas Series to numpy arrays
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
        
    # Reshape y if needed
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.ravel()
        
    return X, y

def get_model_name(model_class: Any) -> str:
    """Get the name of a model class."""
    try:
        if isinstance(model_class, str):
            return model_class
        elif hasattr(model_class, "__name__"):
            return str(model_class.__name__)
        elif hasattr(model_class, "__class__"):
            return str(model_class.__class__.__name__)
        else:
            return str(model_class)
    except Exception as e:
        logger.warning(f"Error getting model name: {e}")
        return str(model_class)