# examples/anomaly_detection_example.py

"""
Anomaly Detection Example using LazyAnomalyDetector from lazypredict.

This script demonstrates how to use LazyAnomalyDetector to automatically fit and evaluate
multiple anomaly detection models on a synthetic dataset.
"""

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

from lazypredict.estimators import LazyAnomalyDetector
from lazypredict.metrics import AnomalyDetectionMetrics
from lazypredict.utils.backend import Backend

# Initialize the backend (pandas is default)
Backend.initialize_backend(use_gpu=False)

def main():
    # Create a synthetic dataset with outliers
    X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5, random_state=42)
    outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
    X = np.vstack((X, outliers))
    y = np.hstack((np.zeros(300), np.ones(20)))  # 0: normal, 1: anomaly

    X = pd.DataFrame(X, columns=['feature1', 'feature2'])
    y = pd.Series(y, name='anomaly')

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize LazyAnomalyDetector
    detector = LazyAnomalyDetector(
        verbose=1,
        ignore_warnings=False,
        random_state=42,
        use_gpu=False,
        mlflow_logging=False,
    )

    # Fit models and get results
    results = detector.fit(X_train, X_test, y_test)

    # Display results
    print("Anomaly Detection Model Evaluation Results:")
    print(results)

    # Access trained models
    models = detector.models

if __name__ == "__main__":
    main()
