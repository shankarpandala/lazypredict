# examples/clustering_example.py

"""
Clustering Example using LazyClusterer from lazypredict.

This script demonstrates how to use LazyClusterer to automatically fit and evaluate
multiple clustering models on the Iris dataset.
"""

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
from sklearn.datasets import load_iris

from lazypredict.estimators import LazyClusterer
from lazypredict.metrics import ClusteringMetrics
from lazypredict.utils.backend import Backend

# Initialize the backend (pandas is default)
Backend.initialize_backend(use_gpu=False)

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Initialize LazyClusterer
    clusterer = LazyClusterer(
        verbose=1,
        ignore_warnings=False,
        random_state=42,
        use_gpu=False,
        mlflow_logging=False,
    )

    # Fit models and get results
    results = clusterer.fit(X)

    # Display results
    print("Clustering Model Evaluation Results:")
    print(results)

    # Access trained models
    models = clusterer.models

if __name__ == "__main__":
    main()
