# examples/classification_example.py

"""
Classification Example using LazyClassifier from lazypredict.

This script demonstrates how to use LazyClassifier to automatically fit and evaluate
multiple classification models on the Iris dataset.
"""

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from lazypredict.estimators import LazyClassifier
from lazypredict.preprocessing import Preprocessor
from lazypredict.metrics import ClassificationMetrics
from lazypredict.utils.backend import Backend

# Initialize the backend (pandas is default)
Backend.initialize_backend(use_gpu=False)

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize LazyClassifier
    clf = LazyClassifier(
        verbose=1,
        ignore_warnings=False,
        predictions=True,
        random_state=42,
        use_gpu=False,
        mlflow_logging=False,
        explainability=True,
    )

    # Fit models and get results
    results, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Display results
    print("Model Evaluation Results:")
    print(results)

    # Access trained models
    models = clf.models

    # Generate explainability reports (SHAP plots are saved as images)
    # Note: Explainability is enabled in the LazyClassifier initialization

if __name__ == "__main__":
    main()
