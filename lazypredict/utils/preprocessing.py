"""
Preprocessing utilities for lazypredict.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

logger = logging.getLogger("lazypredict.preprocessing")

def get_categorical_cardinality_threshold(
    X: pd.DataFrame, categorical_features: List[str], threshold: int = 10
) -> Tuple[List[str], List[str]]:
    """Split categorical features based on their cardinality.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input data.
    categorical_features : List[str]
        List of categorical feature names.
    threshold : int, default=10
        Cardinality threshold for splitting.
        
    Returns
    -------
    Tuple[List[str], List[str]]
        Low cardinality features and high cardinality features.
    """
    low_cardinality = []
    high_cardinality = []
    
    for feature in categorical_features:
        if X[feature].nunique() < threshold:
            low_cardinality.append(feature)
        else:
            high_cardinality.append(feature)
            
    return low_cardinality, high_cardinality

def create_preprocessor(
    X: pd.DataFrame,
    categorical_threshold: int = 10,
    enable_polynomial_features: bool = True,
    polynomial_degree: int = 2,
) -> ColumnTransformer:
    """Create a preprocessing pipeline for the given data.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input data.
    categorical_threshold : int, default=10
        Cardinality threshold for categorical features.
    enable_polynomial_features : bool, default=True
        Whether to create polynomial features.
    polynomial_degree : int, default=2
        Degree of polynomial features.
        
    Returns
    -------
    ColumnTransformer
        Preprocessor pipeline.
    """
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    transformers = []
    
    # Handle numeric features
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")), 
                ("scaler", StandardScaler())
            ]
        )
        transformers.append(("numeric", numeric_transformer, numeric_features))
        
        # Add polynomial features if enabled
        if enable_polynomial_features and numeric_features:
            poly = PolynomialFeatures(
                degree=polynomial_degree, 
                interaction_only=True, 
                include_bias=False
            )
            transformers.append(("poly", poly, numeric_features))
    
    # Handle categorical features
    if categorical_features:
        # Split categorical features based on cardinality
        low_card, high_card = get_categorical_cardinality_threshold(
            X, categorical_features, threshold=categorical_threshold
        )
        
        # Create transformers for low and high cardinality features
        if low_card:
            low_card_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical_low", low_card_transformer, low_card))
            
        if high_card:
            high_card_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoding", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]
            )
            transformers.append(("categorical_high", high_card_transformer, high_card))
    
    # Create and return the column transformer
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor 