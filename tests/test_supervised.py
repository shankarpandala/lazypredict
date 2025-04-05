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