"""
Preprocessing utilities for lazypredict.
"""
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

logger = logging.getLogger("lazypredict.preprocessing")

def categorical_cardinality_threshold(X: Union[pd.DataFrame, np.ndarray], threshold: int = 10) -> List[int]:
    """Identify categorical columns based on cardinality.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    threshold : int, optional (default=10)
        Maximum number of unique values to consider a column categorical.
        
    Returns
    -------
    List[int]
        List of column indices that should be treated as categorical.
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
        
    categorical_mask = []
    
    for i in range(X_values.shape[1]):
        col = X_values[:, i]
        try:
            # Check if column is numeric
            if np.issubdtype(col.dtype, np.number):
                # Handle missing values safely for numeric data
                not_nan_mask = ~np.isnan(col)
                unique_values = np.unique(col[not_nan_mask])
            else:
                # For non-numeric data, just count unique values
                # Treat None and np.nan as the same value
                mask = ~pd.isnull(col)
                unique_values = np.unique(col[mask])
                
            if len(unique_values) <= threshold:
                categorical_mask.append(i)
        except (TypeError, ValueError):
            # If we encounter any errors, assume it's categorical
            categorical_mask.append(i)
            
    return categorical_mask

def create_preprocessor(X: Union[pd.DataFrame, np.ndarray]) -> ColumnTransformer:
    """Create a preprocessor for the given data.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
        
    Returns
    -------
    ColumnTransformer
        Preprocessor that handles numerical and categorical features.
    """
    # Convert to numpy if DataFrame
    if isinstance(X, pd.DataFrame):
        column_names = X.columns
        X_values = X.values
    else:
        X_values = X
        column_names = [f"feature_{i}" for i in range(X_values.shape[1])]
    
    # Identify categorical columns
    categorical_mask = categorical_cardinality_threshold(X_values)
    categorical_indices = categorical_mask
    
    # Create list of numeric indices (all columns not in categorical_mask)
    numerical_indices = [i for i in range(X_values.shape[1]) if i not in categorical_indices]
    
    # Create transformers
    transformers = []
    
    if numerical_indices:
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        transformers.append(('num', numeric_transformer, numerical_indices))
    
    if categorical_indices:
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]
        )
        transformers.append(('cat', categorical_transformer, categorical_indices))
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor 