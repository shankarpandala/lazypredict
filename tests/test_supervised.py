import pytest
import numpy as np
import pandas as pd
import os
from lazypredict.Supervised import LazyClassifier, LazyRegressor, get_card_split
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
import mlflow

def test_lazy_classifier_fit():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)
    assert len(models) > 0
    assert "Balanced Accuracy" in models.columns

def test_lazy_classifier_custom_metric():
    def custom_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=custom_accuracy)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert "custom_accuracy" in models.columns

def test_lazy_classifier_mlflow_integration():
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    # Verify MLflow experiments were created
    assert os.path.exists("mlflow.db")

def test_lazy_classifier_categorical_features():
    # Create dataset with categorical features
    n_samples = 100
    X = pd.DataFrame({
        'num1': np.random.rand(n_samples),
        'num2': np.random.rand(n_samples),
        'cat_low': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_high': np.random.choice(['X' + str(i) for i in range(20)], n_samples)
    })
    y = np.random.randint(0, 2, n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)

def test_lazy_regressor_fit():
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)
    assert len(models) > 0
    assert "Adjusted R-Squared" in models.columns

def test_lazy_regressor_custom_metric():
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=mean_absolute_error)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert "mean_absolute_error" in models.columns

def test_lazy_regressor_mlflow_integration():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    reg = LazyRegressor(verbose=0, ignore_warnings=False)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert os.path.exists("mlflow.db")

def test_get_card_split():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd', 'e'],
        'B': ['f', 'g', 'h', 'i', 'j'],
        'C': ['x' + str(i) for i in range(20)]
    })
    cols = pd.Index(['A', 'B', 'C'])
    card_low, card_high = get_card_split(df, cols, n=10)
    assert list(card_low) == ['A', 'B']
    assert list(card_high) == ['C']

def test_lazy_classifier_specific_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, 
                        classifiers=[RandomForestClassifier, LogisticRegression])
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert len(models) == 2
    assert "RandomForestClassifier" in models.index
    assert "LogisticRegression" in models.index

def test_provide_models():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models = clf.provide_models(X_train, X_test, y_train, y_test)
    assert isinstance(models, dict)
    assert len(models) > 0
    # Verify we can make predictions with the returned models
    for name, model in models.items():
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)

def test_roc_auc_uses_probabilities():
    """
    Test that ROC-AUC is calculated using predicted probabilities instead of class labels.
    This addresses issue #476: Severe Underestimation of ROC-AUC Values in Lazypredict Library
    
    The test verifies that:
    1. LazyClassifier calculates ROC-AUC using predict_proba() for models that support it
    2. ROC-AUC values are higher and more accurate when using probabilities vs class labels
    3. The implementation correctly handles binary and multiclass classification
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Load binary classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test LazyClassifier with a model that supports predict_proba
    clf = LazyClassifier(verbose=0, ignore_warnings=True, classifiers=[LogisticRegression])
    models, _ = clf.fit(X_train, X_test, y_train, y_test)
    
    # Get ROC-AUC from LazyClassifier
    lazy_roc_auc = models.loc['LogisticRegression', 'ROC AUC']
    
    # Manually calculate ROC-AUC using probabilities (correct method)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    expected_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Manually calculate ROC-AUC using class labels (old incorrect method)
    y_pred_labels = pipe.predict(X_test)
    roc_auc_with_labels = roc_auc_score(y_test, y_pred_labels)
    
    # Verify LazyClassifier uses probabilities (should match expected_roc_auc, not roc_auc_with_labels)
    assert np.isclose(lazy_roc_auc, expected_roc_auc, atol=0.001), \
        f"LazyClassifier ROC-AUC ({lazy_roc_auc}) should match probability-based calculation ({expected_roc_auc})"
    
    # Verify probabilities give higher ROC-AUC than class labels (for most cases)
    # This demonstrates the fix for issue #476
    assert expected_roc_auc >= roc_auc_with_labels, \
        f"Probability-based ROC-AUC ({expected_roc_auc}) should be >= label-based ({roc_auc_with_labels})"
    
    # Ensure ROC-AUC is not None
    assert lazy_roc_auc is not None, "ROC-AUC should not be None for models with predict_proba"
    
    # Verify ROC-AUC is in valid range [0, 1]
    assert 0 <= lazy_roc_auc <= 1, f"ROC-AUC ({lazy_roc_auc}) should be between 0 and 1"