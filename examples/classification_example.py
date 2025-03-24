#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive example script for LazyClassifier demonstrating all features:
1. Basic model training and comparison
2. MLflow integration for experiment tracking
3. Hyperparameter optimization with Optuna
4. Custom metric evaluation
5. Specific model selection
6. Model predictions access
"""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    # Try importing from the new structure
    from lazypredict.models.classification import LazyClassifier

    print("Successfully imported from new structure")
except ImportError:
    try:
        # Try importing from old structure for backward compatibility
        from lazypredict.Supervised import LazyClassifier

        print(
            "Successfully imported from old structure (backward compatibility)"
        )
    except ImportError:
        raise ImportError(
            "Failed to import LazyClassifier from either location"
        )


def custom_f1_metric(y_true, y_pred):
    """Example custom metric function"""
    return f1_score(y_true, y_pred, average="weighted")


def main():
    # Load data
    data = load_iris()
    feature_names = data.feature_names
    X = pd.DataFrame(data.data, columns=feature_names)
    y = pd.Series(data.target, name="target")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n1. Basic Classification Example:")
    # Basic classification with all models
    clf = LazyClassifier(
        verbose=1, ignore_warnings=True, random_state=42, predictions=True
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print("\nModel Comparison:")
    print(models)

    print("\n2. MLflow Integration Example:")
    # Create a temporary directory for MLflow tracking
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow_uri = f"file://{tmp_dir}"
        print(f"MLflow tracking URI: {mlflow_uri}")

        # Run with MLflow tracking
        clf_mlflow = LazyClassifier(
            verbose=1, ignore_warnings=True, random_state=42
        )
        models_mlflow, _ = clf_mlflow.fit(
            X_train, X_test, y_train, y_test, mlflow_tracking_uri=mlflow_uri
        )

    print("\n3. Hyperparameter Optimization Example:")
    # Example using RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier

    clf_opt = LazyClassifier(verbose=1, ignore_warnings=True, random_state=42)
    best_params = clf_opt.fit_optimize(
        X_train, y_train, RandomForestClassifier
    )
    print("Best RandomForest Parameters:", best_params)

    print("\n4. Custom Metric Example:")
    # Using custom F1 metric
    clf_custom = LazyClassifier(
        verbose=1,
        ignore_warnings=True,
        random_state=42,
        custom_metric=custom_f1_metric,
    )
    models_custom, _ = clf_custom.fit(X_train, X_test, y_train, y_test)
    print("\nModels with Custom Metric:")
    print(models_custom)

    print("\n5. Specific Models Example:")
    # Using only specific models
    specific_models = ["RandomForestClassifier", "LogisticRegression"]
    clf_specific = LazyClassifier(
        verbose=1,
        ignore_warnings=True,
        random_state=42,
        classifiers=specific_models,
    )
    models_specific, _ = clf_specific.fit(X_train, X_test, y_train, y_test)
    print("\nSpecific Models Comparison:")
    print(models_specific)

    print("\n6. Accessing Predictions Example:")
    # Get predictions from the best model
    best_model_name = models.iloc[0]["Model"]
    trained_models = clf.provide_models(X_train, X_test, y_train, y_test)
    best_model = trained_models[best_model_name]
    y_pred = best_model.predict(X_test)
    print(f"\nPredictions from best model ({best_model_name}):")
    print("First 5 predictions:", y_pred[:5])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
