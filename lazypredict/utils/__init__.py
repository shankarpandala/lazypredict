"""
Utility functions for lazypredict.
"""
import logging
from typing import Any
import numpy as np
import pandas as pd

from .base import BaseLazy
from .gpu import (
    get_best_model,
    get_cpu_model,
    get_gpu_model,
    is_cuml_available,
    is_gpu_available,
)
from .mlflow_utils import (
    configure_mlflow,
    end_run,
    log_artifacts,
    log_dataframe,
    log_metric,
    log_model,
    log_model_performance,
    log_params,
    start_run,
)
from .preprocessing import (
    categorical_cardinality_threshold,
    create_preprocessor,
)

logger = logging.getLogger("lazypredict.utils")

def get_model_name(model_class: Any) -> str:
    """Get the name of a model class.
    
    Parameters
    ----------
    model_class : Any
        The model class.
        
    Returns
    -------
    str
        Name of the model class.
    """
    try:
        if hasattr(model_class, "__name__"):
            return str(model_class.__name__)
        elif hasattr(model_class, "__class__"):
            return str(model_class.__class__.__name__)
        else:
            return str(model_class)
    except:
        # In case of any errors, return the string representation
        return "SomeModel"

def check_X_y(X, y):
    """
    Check and prepare X and y for model training.
    
    Converts pandas DataFrames to numpy arrays if needed.
    
    Parameters
    ----------
    X : array-like
        Features
        
    y : array-like
        Target
        
    Returns
    -------
    X, y : tuple
        Prepared data arrays
    """
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

__all__ = [
    # Base
    "BaseLazy",
    
    # GPU
    "get_best_model",
    "get_cpu_model",
    "get_gpu_model", 
    "is_cuml_available",
    "is_gpu_available",
    
    # MLflow
    "configure_mlflow",
    "end_run",
    "log_artifacts",
    "log_dataframe",
    "log_metric",
    "log_model",
    "log_model_performance",
    "log_params",
    "start_run",
    
    # Preprocessing
    "create_preprocessor",
    "categorical_cardinality_threshold",
    
    # Common utils
    "get_model_name",
    "check_X_y",
]

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) 