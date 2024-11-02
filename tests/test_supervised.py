import pytest
import numpy as np
import pandas as pd
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def test_lazy_classifier_fit():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)

def test_lazy_classifier_provide_models():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    clf.fit(X_train, X_test, y_train, y_test)
    models = clf.provide_models(X_train, X_test, y_train, y_test)
    assert isinstance(models, dict)

def test_lazy_regressor_fit():
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)

def test_lazy_regressor_provide_models():
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    reg.fit(X_train, X_test, y_train, y_test)
    models = reg.provide_models(X_train, X_test, y_train, y_test)
    assert isinstance(models, dict)