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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures

logger = logging.getLogger("lazypredict.preprocessing")

def categorical_cardinality_threshold(data, columns, threshold=10):
    """Split categorical columns based on cardinality threshold."""
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
    
    if not isinstance(columns, (list, tuple)):
        columns = list(columns)
    if not columns:
        return [], []
        
    # Calculate cardinality for each column
    cardinality = {}
    for col in columns:
        if col in data.columns:
            unique_vals = np.unique(data[col])
            cardinality[col] = len(unique_vals[~pd.isnull(unique_vals)])
        else:
            cardinality[col] = 0
            
    # Split based on threshold
    low_card = [col for col in columns if cardinality[col] <= threshold]
    high_card = [col for col in columns if cardinality[col] > threshold]
    
    return low_card, high_card

# Alias for backward compatibility
get_card_split = categorical_cardinality_threshold

def create_preprocessor(data, enable_polynomial_features=True, return_type='pipeline'):
    """Create a preprocessor for mixed data types.
    
    Parameters
    ----------
    data : array-like or pd.DataFrame
        Input data to determine feature types
    enable_polynomial_features : bool, optional (default=True)
        Whether to include polynomial features in the preprocessing pipeline
    return_type : str, optional (default='pipeline')
        Type of transformer to return. One of:
        - 'pipeline': Return full Pipeline including polynomial features
        - 'column': Return only ColumnTransformer
        
    Returns
    -------
    sklearn.base.BaseEstimator
        Preprocessor object (either Pipeline or ColumnTransformer)
    """
    # Import required classes to ensure they're available in the local scope
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    # Convert numpy array to DataFrame for type detection
    if isinstance(data, np.ndarray):
        # Create a DataFrame with feature column names
        column_names = [f"feature_{i}" for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=column_names)
    elif not isinstance(data, pd.DataFrame):
        # Handle other array-like inputs
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Error converting data to DataFrame: {e}. Using empty preprocessor.")
            # Return a simple scaler as fallback
            return StandardScaler()
    
    # Get column types
    numeric_features = list(data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns)
    categorical_features = list(data.select_dtypes(include=['object', 'category']).columns)
    
    transformers = []
    
    # Add numeric transformer if there are numeric features
    if numeric_features:
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_pipeline, numeric_features))
    
    # Add categorical transformers if there are categorical features
    if categorical_features:
        low_card, high_card = categorical_cardinality_threshold(data, categorical_features)
        if low_card:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_pipeline, low_card))
        
        # For high cardinality categorical features, use OrdinalEncoder
        if high_card:
            high_card_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('high_cat', high_card_pipeline, high_card))
    
    # If data is a numpy array or only has one type of feature,
    # use numeric pipeline on all features
    if not transformers:
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('all', numeric_pipeline, list(range(data.shape[1]))))
    
    # Create the preprocessor without polynomial features first
    base_preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    
    # Return based on type requested
    if return_type == 'column' or not enable_polynomial_features:
        return base_preprocessor
    
    # Add polynomial features if enabled and there are numeric features
    if numeric_features or not transformers:
        return Pipeline([
            ('preprocessor', base_preprocessor),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
    
    return base_preprocessor