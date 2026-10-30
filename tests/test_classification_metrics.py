"""Classifier metrics: CV scorers (issue #584) and ROC AUC / NaN handling (issue #585)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from lazypredict.Supervised import LazyClassifier, LazyRegressor

CV_COLUMNS = [
    "Accuracy CV Mean", "Balanced Accuracy CV Mean", "F1 Score CV Mean",
    "Precision CV Mean", "Recall CV Mean", "ROC AUC CV Mean",
]


@pytest.fixture(scope="module")
def binary_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def _fit(clf, split):
    X_train, X_test, y_train, y_test = split
    return clf.fit(X_train, X_test, y_train, y_test)[0]


def test_cv_metrics_for_models_without_predict_proba(binary_split):
    clf = LazyClassifier(
        classifiers=["RidgeClassifier", "LinearSVC", "LogisticRegression"], cv=3, n_jobs=1
    )
    scores = _fit(clf, binary_split)
    # decision_function is enough for binary ROC AUC, as on the test set
    assert scores[CV_COLUMNS].notna().all().all(), scores[CV_COLUMNS]
    assert (scores["ROC AUC CV Mean"] > 0.9).all()


def test_multiclass_cv_roc_auc_needs_probabilities():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    clf = LazyClassifier(classifiers=["RidgeClassifier", "LogisticRegression"], cv=3, n_jobs=1)
    scores = clf.fit(X[:200], X[200:], y[:200], y[200:])[0]
    # RidgeClassifier has no predict_proba: only its ROC AUC column is blank
    assert np.isnan(scores.loc["RidgeClassifier", "ROC AUC CV Mean"])
    assert scores.loc["RidgeClassifier", CV_COLUMNS[:-1]].notna().all()
    assert scores.loc["LogisticRegression", CV_COLUMNS].notna().all()


def test_multiclass_roc_auc_when_test_split_misses_a_class():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 4)
    y = np.digitize(X[:, 0], [-0.5, 0.5])
    keep = y[300:] != 2
    clf = LazyClassifier(classifiers=["LogisticRegression", "RandomForestClassifier"])
    scores = clf.fit(X[:300], X[300:][keep], y[:300], y[300:][keep])[0]
    assert scores["ROC AUC"].notna().all()
    assert (scores["ROC AUC"] > 0.9).all()


def test_metric_columns_stay_numeric_when_a_metric_is_missing(binary_split):
    def failing_metric(y_true, y_pred):
        raise RuntimeError("boom")

    X_train, X_test, y_train, y_test = binary_split
    # A single-class test set makes ROC AUC undefined.
    only_ones = y_test == 1
    clf = LazyClassifier(classifiers=["LogisticRegression"], custom_metric=failing_metric)
    scores = clf.fit(X_train, X_test[only_ones], y_train, y_test[only_ones])[0]
    assert np.isnan(scores.loc["LogisticRegression", "ROC AUC"])
    assert np.isnan(scores.loc["LogisticRegression", "failing_metric"])
    for column in ["ROC AUC", "failing_metric"]:
        assert pd.api.types.is_float_dtype(scores[column]), scores.dtypes


def test_regressor_cv_columns(binary_split):
    reg = LazyRegressor(regressors=["Ridge"], cv=3, n_jobs=1)
    scores = _fit(reg, binary_split)
    assert scores[["R-Squared CV Mean", "RMSE CV Mean"]].notna().all().all()


def test_weighted_ovr_auc_matches_scikit_learn():
    from sklearn.metrics import roc_auc_score

    from lazypredict.Supervised import _weighted_ovr_roc_auc

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    y[:4] = np.arange(4)
    proba = rng.dirichlet(np.ones(4), 200)
    expected = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    assert _weighted_ovr_roc_auc(y, proba, np.arange(4)) == pytest.approx(expected)
