"""Public estimator coverage for categorical pandas dtypes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lazypredict.Supervised import LazyClassifier, LazyRegressor
from lazypredict.preprocessing import get_categorical_encoder


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


@pytest.mark.parametrize("missing_in_train", [False, True])
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_numeric_category_identity_with_missing_values(missing_in_train, encoder, task):
    """Missing entries must not change the identity of integer-valued categories."""
    train_values = [9, 17, None] if missing_in_train else [9, 17]
    test_values = [9, 17] if missing_in_train else [9, 17, None]
    X_train = pd.DataFrame({"value": pd.Series(train_values * 4, dtype="category")})
    X_test = pd.DataFrame({"value": pd.Series(test_values * 4, dtype="category")})
    train_before, test_before = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train = np.tile([0, 1, 0] if missing_in_train else [0, 1], 4)
    y_test = np.tile([0, 1] if missing_in_train else [0, 1, 0], 4)
    known = X_test["value"].notna().to_numpy()
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
    np.testing.assert_array_equal(predictions[model_name][known], y_test[known])
    np.testing.assert_array_equal(estimator.predict(X_test, model_name)[known], y_test[known])
    pd.testing.assert_frame_equal(X_train, train_before)
    pd.testing.assert_frame_equal(X_test, test_before)


def test_mixed_type_categories_remain_distinct():
    """Normalization must not collapse distinct numeric and string labels."""
    X = pd.DataFrame({"value": pd.Series([1, "1", None], dtype="category")})
    before = X.copy(deep=True)
    normalized = get_categorical_encoder().named_steps["to_object"].fit_transform(X)

    assert normalized["value"].iloc[0] == 1
    assert normalized["value"].iloc[1] == "1"
    assert normalized["value"].iloc[0] != normalized["value"].iloc[1]
    assert pd.isna(normalized["value"].iloc[2])
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize(
    "values,dtype",
    [
        (["red", "blue", None], "string"),
        (["red", "blue", None], "category"),
        ([0, 1, 0], "category"),
        ([0.5, 1.5, 0.5], "category"),
        ([False, True, False], "category"),
        ([0, 1, None], "category"),
        ([0.5, 1.5, None], "category"),
        ([False, True, None], "category"),
    ],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mixed", [False, True])
def test_categorical_extension_values_with_numeric_features(values, dtype, encoder, task, mixed):
    """Non-string categories and pandas missing values remain valid inputs."""
    X_train = pd.DataFrame({"category": pd.Series(values * 4, dtype=dtype)})
    if mixed:
        X_train["constant"] = 0.0
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
    np.testing.assert_array_equal(estimator.predict(X_test, model_name), y_test)
    preprocessor = estimator.models[model_name].named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    assert any("category" in name for name in feature_names)
    assert len(feature_names) == preprocessor.transform(X_test).shape[1]
    pd.testing.assert_frame_equal(X_train, train_before)
    pd.testing.assert_frame_equal(X_test, test_before)
