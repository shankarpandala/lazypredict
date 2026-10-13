"""Public estimator coverage for categorical pandas dtypes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lazypredict.Supervised import LazyClassifier, LazyRegressor


@pytest.mark.parametrize("dtype", ["object", "string", "category", None])
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_categorical_dtypes_keep_predictive_features(dtype, encoder, mixed, task):
    """Extension dtypes must retain the same signal as object columns."""
    X_train = pd.DataFrame({"color": pd.Series(["red", "blue"] * 6, dtype=dtype)})
    X_test = pd.DataFrame({"color": pd.Series(["red", "blue"] * 2, dtype=dtype)})
    if mixed:
        X_train["constant"] = 0.0
        X_test["constant"] = 0.0
    train_before = X_train.copy(deep=True)
    test_before = X_test.copy(deep=True)
    y_train = np.array([0, 1] * 6)
    y_test = np.array([0, 1] * 2)

    kwargs = {"categorical_encoder": encoder, "predictions": True, "n_jobs": 1}
    if task == "classification":
        estimator = LazyClassifier(classifiers=[DecisionTreeClassifier], **kwargs)
        model_name = "DecisionTreeClassifier"
        score_name = "Accuracy"
    else:
        estimator = LazyRegressor(regressors=[DecisionTreeRegressor], **kwargs)
        model_name = "DecisionTreeRegressor"
        score_name = "R-Squared"

    scores, predictions = estimator.fit(X_train, X_test, y_train, y_test)

    assert not estimator.errors
    assert scores.index.tolist() == [model_name]
    assert scores.loc[model_name, score_name] == pytest.approx(1.0)
    np.testing.assert_array_equal(predictions[model_name], y_test)
    np.testing.assert_array_equal(estimator.predict(X_test, model_name), y_test)
    feature_names = estimator.models[model_name].named_steps["preprocessor"].get_feature_names_out()
    assert any("color" in name for name in feature_names)
    assert len(feature_names) == (2 if encoder == "onehot" else 1) + int(mixed)
    pd.testing.assert_frame_equal(X_train, train_before)
    pd.testing.assert_frame_equal(X_test, test_before)


@pytest.mark.parametrize(
    "values,dtype",
    [
        (["red", "blue", None], "string"),
        (["red", "blue", None], "category"),
        ([0, 1, 0], "category"),
        ([0.5, 1.5, 0.5], "category"),
        ([False, True, False], "category"),
    ],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_categorical_extension_values_with_numeric_features(values, dtype, encoder, task):
    """Non-string categories and pandas missing values remain valid inputs."""
    X_train = pd.DataFrame({"category": pd.Series(values * 4, dtype=dtype), "constant": 0.0})
    X_test = X_train.iloc[:6].copy()
    train_before = X_train.copy(deep=True)
    test_before = X_test.copy(deep=True)
    y_train = np.array([0, 1, 0] * 4)
    y_test = y_train[:6]

    kwargs = {"categorical_encoder": encoder, "predictions": True, "n_jobs": 1}
    if task == "classification":
        estimator = LazyClassifier(classifiers=[DecisionTreeClassifier], **kwargs)
        model_name = "DecisionTreeClassifier"
    else:
        estimator = LazyRegressor(regressors=[DecisionTreeRegressor], **kwargs)
        model_name = "DecisionTreeRegressor"

    scores, predictions = estimator.fit(X_train, X_test, y_train, y_test)

    assert not estimator.errors
    assert scores.index.tolist() == [model_name]
    np.testing.assert_array_equal(predictions[model_name], y_test)
    pd.testing.assert_frame_equal(X_train, train_before)
    pd.testing.assert_frame_equal(X_test, test_before)
