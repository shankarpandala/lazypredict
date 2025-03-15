"""
Utility functions for lazypredict.
"""

import numpy as np
import pandas as pd

def get_model_name(model):
    """
    Get the name of a model.
    
    Parameters
    ----------
    model : object or class or str
        Model instance, class, or name
        
    Returns
    -------
    name : str
        Name of the model
    """
    if isinstance(model, str):
        return model
    elif hasattr(model, "__name__"):
        return model.__name__
    else:
        return model.__class__.__name__

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