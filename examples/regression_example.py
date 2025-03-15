#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for running LazyRegressor on a standard dataset.

This script demonstrates:
1. Loading a dataset
2. Splitting into train/test
3. Training multiple models with LazyRegressor
4. Retrieving and using the trained models
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

try:
    # Try importing from the new structure
    from lazypredict.models.regression import LazyRegressor
    print("Successfully imported from new structure")
except ImportError:
    try:
        # Try importing from old structure for backward compatibility
        from lazypredict.Supervised import LazyRegressor
        print("Successfully imported from old structure (backward compatibility)")
    except ImportError:
        raise ImportError("Failed to import LazyRegressor from either location")

def main():
    # Load data
    data = load_diabetes()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and configure the regressor
    reg = LazyRegressor(verbose=1, ignore_warnings=True)
    
    # Fit on the data
    print("Fitting models...")
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\nModel Comparison:")
    print(models)
    
    # Get the best performing model
    best_model_name = models.iloc[0].name
    print(f"\nBest performing model: {best_model_name}")
    
    # Retrieve the trained models
    trained_models = reg.provide_models(X_train, X_test, y_train, y_test)
    
    # Use the best model to make predictions
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics manually
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Best model R² (manual calculation): {r2:.4f}")
    print(f"Best model MAE (manual calculation): {mae:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}") 