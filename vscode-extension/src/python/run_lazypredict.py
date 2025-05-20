#!/usr/bin/env python
"""
Script to run LazyPredict models on a dataset.
This script is called by the VS Code extension to perform analysis.
"""
import sys
import json
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import traceback

def run_lazypredict(file_path, target_column, problem_type='classification', test_size=0.2, random_state=42, custom_models=None):
    """
    Run LazyPredict on the provided dataset.
    
    Args:
        file_path (str): Path to the CSV or Excel file
        target_column (str): Name of the target column
        problem_type (str): Type of problem - 'classification' or 'regression'
        test_size (float): Size of test set (between 0 and 1)
        random_state (int): Random state for reproducibility
        custom_models (list): Optional list of specific models to run
        
    Returns:
        dict: Results of the analysis
    """
    try:
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file format. Please use CSV or Excel files."}

        # Check if target column exists
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in the dataset."}

        # Split data into features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle non-numeric columns (one-hot encoding)
        X = pd.get_dummies(X)
        
        # If there are NaN values, fill them
        X = X.fillna(X.mean())
        
        # Check if target is categorical for classification problems
        if problem_type == 'classification' and not pd.api.types.is_categorical_dtype(y):
            # For classification, try to convert to categorical if it's not already
            y = y.astype('category')
        
        # Run LazyPredict
        if problem_type == 'classification':
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=random_state)
            models, predictions = clf.fit(X, y, test_size=test_size)
        else:  # regression
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, random_state=random_state)
            models, predictions = reg.fit(X, y, test_size=test_size)
        
        # Format the results
        models_df = models.reset_index().rename(columns={'index': 'name'})
        
        # Handle NaN values in the results
        models_df = models_df.fillna('N/A')
        
        # Convert DataFrame to list of dictionaries
        models_list = models_df.to_dict('records')
        
        # Identify the best model
        if problem_type == 'classification':
            best_metric = 'Accuracy'
        else:
            best_metric = 'R-Squared'
        
        best_model = models_df.iloc[0]['name']  # LazyPredict sorts by best performance
        
        # Format final results
        results = {
            "models": models_list,
            "bestModel": best_model,
            "problemType": problem_type,
            "logs": f"Analysis completed successfully. {len(models_list)} models were trained and evaluated."
        }
        
        return results
    
    except Exception as e:
        error_message = f"Error: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_message}

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Missing required arguments. Usage: python run_lazypredict.py <file_path> <target_column> <problem_type> [<test_size>] [<random_state>]"}))
        sys.exit(1)
    
    file_path = sys.argv[1]
    target_column = sys.argv[2]
    
    problem_type = sys.argv[3] if len(sys.argv) > 3 else 'classification'
    test_size = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    random_state = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    
    # Parse custom models if provided (comma-separated list)
    custom_models = sys.argv[6].split(',') if len(sys.argv) > 6 else None
    
    # Run LazyPredict and print results as JSON
    results = run_lazypredict(file_path, target_column, problem_type, test_size, random_state, custom_models)
    print(json.dumps(results, default=str))  # default=str handles non-serializable objects