"""Prediction rows must remain aligned with their input samples."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lazypredict import LazyClassifier, LazyRegressor


@pytest.fixture(params=["classifier", "regressor"])
def estimator(request):
    if request.param == "classifier":
        return LazyClassifier(
            classifiers=[DecisionTreeClassifier], predictions=True, n_jobs=1
        )
    return LazyRegressor(
        regressors=[DecisionTreeRegressor], predictions=True, n_jobs=1
    )


def fit_data(estimator, index=None, as_array=False):
    X_train = pd.DataFrame({"signal": [0, 1, 2, 3, 0, 1, 2, 3]})
    X_test = pd.DataFrame({"signal": [3, 0, 2, 1]}, index=index)
    y_train = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_test = np.array([1, 0, 1, 0])
    if as_array:
        X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    scores, predictions = estimator.fit(X_train, X_test, y_train, y_test)
    return X_test, y_test, scores, predictions


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([2, 0, 3, 1], name="sample"),
        pd.Index(["d", "a", "c", "b"], name="sample"),
        pd.Index(["same", "other", "same", "last"], name="sample"),
        pd.MultiIndex.from_tuples(
            [("b", 2), ("a", 1), ("b", 1), ("a", 2)],
            names=["group", "sample"],
        ),
    ],
    ids=["shuffled-integers", "strings", "duplicates", "multiindex"],
)
def test_predictions_preserve_test_index(estimator, index):
    X_test, expected, scores, predictions = fit_data(estimator, index=index)

    assert len(scores) == 1
    assert not estimator.errors
    np.testing.assert_array_equal(predictions.iloc[:, 0].to_numpy(), expected)
    pd.testing.assert_index_equal(predictions.index, X_test.index)
    # Assignment aligns by label; a fresh RangeIndex can silently shuffle rows.
    combined = X_test.assign(prediction=predictions.iloc[:, 0])
    np.testing.assert_array_equal(combined["prediction"].to_numpy(), expected)


@pytest.mark.parametrize("as_array", [False, True], ids=["dataframe", "numpy"])
def test_predictions_keep_default_index(estimator, as_array):
    _, expected, scores, predictions = fit_data(estimator, as_array=as_array)

    assert len(scores) == 1
    assert not estimator.errors
    pd.testing.assert_index_equal(predictions.index, pd.RangeIndex(4))
    np.testing.assert_array_equal(predictions.iloc[:, 0].to_numpy(), expected)


def test_predictions_disabled_stays_empty(estimator):
    estimator.predictions = False
    _, _, scores, predictions = fit_data(estimator, index=[2, 0, 3, 1])

    assert len(scores) == 1
    assert not estimator.errors
    pd.testing.assert_frame_equal(predictions, pd.DataFrame())


def test_dataframe_train_numpy_test_keeps_default_index(estimator):
    X_train = pd.DataFrame([[0], [1], [2], [3]] * 2)
    X_test = np.array([[3], [0], [2], [1]])
    y_train = np.array([0, 0, 1, 1] * 2)
    expected = np.array([1, 0, 1, 0])

    scores, predictions = estimator.fit(X_train, X_test, y_train, expected)

    assert len(scores) == 1
    assert not estimator.errors
    pd.testing.assert_index_equal(predictions.index, pd.RangeIndex(4))
    np.testing.assert_array_equal(predictions.iloc[:, 0].to_numpy(), expected)


def test_no_models_keeps_predictions_empty(estimator):
    if isinstance(estimator, LazyClassifier):
        estimator.classifiers = []
    else:
        estimator.regressors = []
    _, _, scores, predictions = fit_data(estimator, index=[2, 0, 3, 1])

    assert scores.empty
    pd.testing.assert_frame_equal(predictions, pd.DataFrame())
