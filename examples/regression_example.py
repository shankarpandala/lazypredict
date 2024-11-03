# examples/regression_example.py

"""
Regression Example using LazyRegressor from lazypredict.

This script demonstrates how to use LazyRegressor to automatically fit and evaluate
multiple regression models on the California Housing dataset.
"""

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from lazypredict.estimators import LazyRegressor
from lazypredict.utils.backend import Backend

# Initialize the backend (pandas is default)
Backend.initialize_backend(use_gpu=False)


def main():
    # Load the California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize LazyRegressor
    reg = LazyRegressor(
        verbose=1,
        ignore_warnings=False,
        predictions=True,
        random_state=42,
        use_gpu=False,
        mlflow_logging=False,
        explainability=True,
    )

    # Fit models and get results
    results, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Display results
    print("Model Evaluation Results:")
    print(results)

    # Access trained models
    models = reg.models

    # Generate explainability reports (SHAP plots are saved as images)
    # Note: Explainability is enabled in the LazyRegressor initialization


if __name__ == "__main__":
    main()
