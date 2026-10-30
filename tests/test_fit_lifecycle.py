"""fit() state, target validation and mixed input types (issues #586, #587, #589)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from lazypredict.Supervised import LazyClassifier, LazyRegressor


@pytest.fixture(scope="module")
def frames():
    X, y = load_breast_cancer(return_X_y=True)
    X = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])
    return train_test_split(X, y, random_state=0)


# -- #586 ---------------------------------------------------------------------

def test_refit_does_not_keep_previous_models(frames):
    clf = LazyClassifier(classifiers=["LogisticRegression", "DecisionTreeClassifier"])
    clf.fit(*frames)
    clf.classifiers = ["DecisionTreeClassifier"]
    clf.fit(*frames)
    assert list(clf.models) == ["DecisionTreeClassifier"]
    assert list(clf.predict(frames[1])) == ["DecisionTreeClassifier"]


def test_refit_clears_previous_errors(frames):
    X_train, X_test, y_train, y_test = frames
    clf = LazyClassifier(classifiers=["LogisticRegression"])
    clf.fit(X_train.assign(c0=np.nan), X_test, y_train, y_test)  # LogisticRegression copes via imputation
    clf.errors["stale"] = RuntimeError("from an earlier run")
    clf.fit(X_train, X_test, y_train, y_test)
    assert "stale" not in clf.errors


def test_pipelines_do_not_share_the_preprocessor(frames):
    clf = LazyClassifier(classifiers=["LogisticRegression", "DecisionTreeClassifier"])
    clf.fit(*frames)
    first, second = (pipe.named_steps["preprocessor"] for pipe in clf.models.values())
    assert first is not second


# -- #587 ---------------------------------------------------------------------

def test_classifier_rejects_continuous_target():
    X, y = load_diabetes(return_X_y=True)
    y = y + 0.5 * np.random.RandomState(0).rand(len(y))  # non-integral values
    with pytest.raises(ValueError, match="LazyRegressor"):
        LazyClassifier(classifiers=["LogisticRegression"]).fit(X[:300], X[300:], y[:300], y[300:])


def test_classifier_warns_for_integer_regression_targets():
    X, y = load_diabetes(return_X_y=True)  # integer-valued regression target
    with pytest.warns(UserWarning, match="LazyRegressor"):
        LazyClassifier(classifiers=["DecisionTreeClassifier"]).fit(X[:300], X[300:], y[:300], y[300:])


def test_classifier_accepts_integral_float_labels():
    X, y = load_breast_cancer(return_X_y=True)
    scores, _ = LazyClassifier(classifiers=["LogisticRegression"]).fit(
        X[:400], X[400:], y[:400].astype(float), y[400:].astype(float)
    )
    assert len(scores) == 1


# -- #589 ---------------------------------------------------------------------

@pytest.mark.parametrize("train_as_array, test_as_array", [(False, True), (True, False), (True, True)])
def test_mixed_numpy_and_dataframe_inputs(frames, train_as_array, test_as_array):
    X_train, X_test, y_train, y_test = frames
    clf = LazyClassifier(classifiers=["LogisticRegression"])
    scores, _ = clf.fit(
        X_train.to_numpy() if train_as_array else X_train,
        X_test.to_numpy() if test_as_array else X_test,
        y_train, y_test,
    )
    assert not clf.errors
    assert scores.loc["LogisticRegression", "Accuracy"] > 0.9


def test_lists_of_lists_are_accepted(frames):
    X_train, X_test, y_train, y_test = frames
    reg = LazyRegressor(regressors=["Ridge"])
    scores, _ = reg.fit(X_train.values.tolist(), X_test.values.tolist(), list(y_train), list(y_test))
    assert len(scores) == 1


def test_predict_and_explain_accept_numpy_after_dataframe_fit(frames):
    X_train, X_test, y_train, y_test = frames
    clf = LazyClassifier(classifiers=["LogisticRegression"])
    clf.fit(X_train, X_test, y_train, y_test)
    np.testing.assert_array_equal(
        clf.predict(X_test.to_numpy(), model_name="LogisticRegression"),
        clf.predict(X_test, model_name="LogisticRegression"),
    )
    importances = clf.explain(X_test.to_numpy(), y_test, n_repeats=2)
    assert list(importances.index[:1]) and set(importances.index) <= set(X_train.columns)


def test_predict_after_load_models_uses_fitted_column_names(frames, tmp_path):
    X_train, X_test, y_train, y_test = frames
    clf = LazyClassifier(classifiers=["LogisticRegression"])
    clf.fit(X_train, X_test, y_train, y_test)
    clf.save_models(str(tmp_path))
    fresh = LazyClassifier()
    fresh.load_models(str(tmp_path))
    preds = fresh.predict(X_test.to_numpy(), model_name="LogisticRegression")
    assert len(preds) == len(X_test)
