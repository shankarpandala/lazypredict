#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive example script for LazyRegressor demonstrating all features:
1. Basic regression model training and comparison
2. MLflow integration for experiment tracking
3. Hyperparameter optimization with Optuna
4. Custom metric evaluation
5. Specific model selection
6. Model accessing and prediction
7. Adjusted R-squared metric
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tempfile
import os

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

def custom_rmse_metric(y_true, y_pred):
    """Example custom metric function - RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # Load data
    data = load_diabetes()
    feature_names = data.feature_names
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n1. Basic Regression Example:")
    # Basic regression with all models
    reg = LazyRegressor(verbose=1, ignore_warnings=True, random_state=42)
    models, _ = reg.fit(X_train, X_test, y_train, y_test)
    print("\nModel Comparison:")
    print(models)
    
    print("\n2. MLflow Integration Example:")
    # Create a temporary directory for MLflow tracking
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow_uri = f"file://{tmp_dir}"
        print(f"MLflow tracking URI: {mlflow_uri}")
        
        # Run with MLflow tracking
        reg_mlflow = LazyRegressor(verbose=1, ignore_warnings=True, random_state=42)
        models_mlflow, _ = reg_mlflow.fit(
            X_train, X_test, y_train, y_test,
            mlflow_tracking_uri=mlflow_uri
        )
        
    print("\n3. Hyperparameter Optimization Example:")
    # Example using RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    reg_opt = LazyRegressor(verbose=1, ignore_warnings=True, random_state=42)
    best_params = reg_opt.fit_optimize(X_train, y_train, RandomForestRegressor)
    print("Best RandomForest Parameters:", best_params)
    
    print("\n4. Custom Metric Example:")
    # Using custom RMSE metric
    reg_custom = LazyRegressor(
        verbose=1,
        ignore_warnings=True,
        random_state=42,
        custom_metric=custom_rmse_metric
    )
    models_custom, _ = reg_custom.fit(X_train, X_test, y_train, y_test)
    print("\nModels with Custom Metric (RMSE):")
    print(models_custom)
    
    print("\n5. Specific Models Example:")
    # Using only specific models
    specific_models = ['RandomForestRegressor', 'LinearRegression']
    reg_specific = LazyRegressor(
        verbose=1,
        ignore_warnings=True,
        random_state=42,
        regressors=specific_models
    )
    models_specific, _ = reg_specific.fit(X_train, X_test, y_train, y_test)
    print("\nSpecific Models Comparison:")
    print(models_specific)
    
    print("\n6. Model Access and Prediction Example:")
    # Get the best model name
    best_model_name = models.iloc[0]['Model']
    
    # Train and get the best model
    trained_models = reg.provide_models(X_train, X_test, y_train, y_test)
    best_model = trained_models[best_model_name]
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nBest model ({best_model_name}) performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("First 5 predictions:", y_pred[:5])
    print("First 5 actual values:", y_test.values[:5])
    
    print("\n7. Metrics Explanation:")
    print("- R-Squared: Standard R² score")
    print("- Adjusted R-Squared: R² adjusted for number of predictors")
    print("- MSE: Mean squared error")
    print("- RMSE: Root mean squared error")
    print("- MAE: Mean absolute error")
    print("- Time Taken: Training time in seconds")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")