"""Selecting models by name or class (issue #583)."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from lazypredict.Supervised import LazyClassifier, LazyRegressor


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 3)
    y = (X[:, 0] > 0).astype(int)
    return X[:60], X[60:], y[:60], y[60:]


def test_single_string_selects_exactly_that_model(binary_data):
    scores, _ = LazyClassifier(classifiers="LinearSVC").fit(*binary_data)
    assert list(scores.index) == ["LinearSVC"]  # not SVC as well


def test_unknown_name_raises_with_suggestion(binary_data):
    with pytest.raises(ValueError, match="RandomForest.*did you mean 'RandomForestClassifier'"):
        LazyClassifier(classifiers=["RandomForest"]).fit(*binary_data)


def test_unknown_regressor_name_raises(binary_data):
    with pytest.raises(ValueError, match="Unknown regressor name"):
        LazyRegressor(regressors=["NotAModel"]).fit(*binary_data)


@pytest.mark.parametrize(
    "selection",
    [
        ["LogisticRegression", DecisionTreeClassifier],
        [DecisionTreeClassifier, "LogisticRegression"],
    ],
)
def test_mixed_names_and_classes(binary_data, selection):
    scores, _ = LazyClassifier(classifiers=selection).fit(*binary_data)
    assert set(scores.index) == {"LogisticRegression", "DecisionTreeClassifier"}


def test_duplicates_are_trained_once(binary_data):
    clf = LazyClassifier(classifiers=["LogisticRegression", LogisticRegression])
    scores, _ = clf.fit(*binary_data)
    assert list(scores.index) == ["LogisticRegression"]


def test_instances_are_rejected(binary_data):
    with pytest.raises(ValueError, match="not instances"):
        LazyRegressor(regressors=[LinearRegression()]).fit(*binary_data)


def test_all_and_empty_selection(binary_data):
    clf = LazyClassifier(classifiers=[])
    scores, _ = clf.fit(*binary_data)
    assert scores.empty
    assert len(LazyClassifier()._get_estimator_list()) > 10
