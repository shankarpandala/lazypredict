"""
Test script to verify MLflow logging functionality in LazyPredict
"""
import os
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

def test_mlflow_logging():
    """Test that models are being logged to MLflow correctly"""
    
    print("=" * 80)
    print("Testing MLflow Model Logging")
    print("=" * 80)
    
    # Set MLflow tracking URI via environment variable (required for LazyPredict)
    os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
    print(f"\n1. Setting MLFLOW_TRACKING_URI: {os.environ['MLFLOW_TRACKING_URI']}")
    
    # Set MLflow experiment
    experiment_name = "lazypredict_test"
    mlflow.set_experiment(experiment_name)
    print(f"\n2. MLflow experiment set to: {experiment_name}")
    
    # Load dataset
    print("\n3. Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    print("\n4. Splitting data (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize LazyClassifier (MLflow enabled via environment variable)
    print("\n5. Initializing LazyClassifier (MLflow logging enabled via MLFLOW_TRACKING_URI)...")
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42
    )
    
    # Train models
    print("\n6. Training models (this will log to MLflow)...")
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    print("\n7. Results:")
    print("=" * 80)
    print(models.head(10))
    print("=" * 80)
    
    print(f"\n✅ Successfully trained {len(models)} models")
    print(f"✅ Best model: {models.index[0]}")
    print(f"✅ Best accuracy: {models['Accuracy'].iloc[0]:.4f}")
    
    # Check MLflow runs
    print("\n8. Checking MLflow runs...")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"✅ Found {len(runs)} runs in MLflow")
        print(f"\nRecent runs:")
        if not runs.empty:
            print(runs[['run_id', 'tags.mlflow.runName', 'metrics.accuracy', 'status']].head(10))
    else:
        print("⚠️ No experiment found")
    
    print("\n" + "=" * 80)
    print("MLflow UI available at: http://127.0.0.1:5000")
    print("=" * 80)
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_mlflow_logging()
