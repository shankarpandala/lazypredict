"""Public estimator coverage for categorical pandas dtypes."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lazypredict.Supervised import LazyClassifier, LazyRegressor
from lazypredict.preprocessing import get_categorical_encoder


@pytest.mark.parametrize(
    "train_values,test_values",
    [
        ([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), None],
         [datetime(2020, 1, 2), datetime(2020, 1, 1), None]),
        ([pd.Timestamp("2020-01-01T00:00:00.000000001"),
          pd.Timestamp("2020-01-01T00:00:00.000000002"), None],
         [np.datetime64("2020-01-01T00:00:00.000000002"),
          np.datetime64("2020-01-01T00:00:00.000000001"), np.datetime64("NaT", "ns")]),
        ([pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-02", tz="UTC"), None],
         [datetime(2020, 1, 2, 2, tzinfo=timezone(timedelta(hours=2))),
          datetime(2020, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))), None]),
        ([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), None],
         [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-01"), None]),
        ([datetime(2500, 1, 1), datetime(2500, 1, 2), None],
         [datetime(2500, 1, 2), datetime(2500, 1, 1), None]),
        ([datetime(2500, 1, 1), datetime(2500, 1, 2), None],
         [np.datetime64("2500-01-02", "us"), np.datetime64("2500-01-01", "us"), None]),
        pytest.param(
            [np.datetime64("12000-01-01", "D"), np.datetime64("12000-01-02", "D"), None],
            [np.datetime64("12000-01-02", "us"), np.datetime64("12000-01-01", "us"), None],
            id="wide-numpy-year",
        ),
        pytest.param(
            [np.datetime64("12000-01-01T00:00:00.000001", "us"),
             np.datetime64("12000-01-01T00:00:00.000002", "us"), None],
            [np.datetime64("12000-01-01T00:00:00.000002", "us"),
             np.datetime64("12000-01-01T00:00:00.000001", "us"), None],
            id="wide-numpy-microseconds",
        ),
        ([datetime(2500, 1, 1, tzinfo=timezone.utc), datetime(2500, 1, 2, tzinfo=timezone.utc), None],
         [datetime(2500, 1, 2, 2, tzinfo=timezone(timedelta(hours=2))),
          datetime(2500, 1, 1, 2, tzinfo=timezone(timedelta(hours=2))), None]),
    ],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mixed", [False, True])
def test_datetime_category_labels_match_equivalent_prediction_values(
    train_values, test_values, encoder, task, mixed
):
    """Equivalent instants retain known labels across datetime representations."""
    categories = pd.Index(train_values[:-1], dtype=object)
    X_train = pd.DataFrame({"signal": pd.Categorical.from_codes(np.tile([0, 1, -1], 4), categories=categories)})
    X_test = pd.DataFrame({"signal": pd.Series(test_values * 3, dtype=object)})
    if mixed:
        X_train["constant"], X_test["constant"] = 2.25, 2.25
    before_train, before_test = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.tile([0, 1, 0], 4), np.tile([1, 0, 0], 3)
    kwargs = {"predictions": True, "n_jobs": 1, "categorical_encoder": encoder}
    if task == "classification":
        model = LazyClassifier(classifiers=[DecisionTreeClassifier], **kwargs)
        name = "DecisionTreeClassifier"
    else:
        model = LazyRegressor(regressors=[DecisionTreeRegressor], **kwargs)
        name = "DecisionTreeRegressor"
    scores, predictions = model.fit(X_train, X_test, y_train, y_test)
    assert not model.errors
    assert scores.index.tolist() == [name]
    np.testing.assert_array_equal(predictions[name], y_test)
    np.testing.assert_array_equal(model.predict(X_test, name), y_test)
    pre = model.models[name].named_steps["preprocessor"]
    encoded_train, encoded_test = pre.transform(X_train), pre.transform(X_test)
    np.testing.assert_array_equal(encoded_train[0], encoded_test[1])
    np.testing.assert_array_equal(encoded_train[1], encoded_test[0])
    assert len(pre.get_feature_names_out()) == encoded_test.shape[1]
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


@pytest.mark.parametrize(
    "train_values,test_values",
    [
        ([pd.Timedelta(days=1), pd.Timedelta(days=2)],
         [timedelta(days=2), timedelta(days=1)]),
        ([pd.Timedelta(days=1), pd.Timedelta(days=2)],
         [np.timedelta64(48, "h"), np.timedelta64(24, "h")]),
        ([pd.Timedelta(microseconds=-2), pd.Timedelta(microseconds=1)],
         [timedelta(microseconds=1), np.timedelta64(-2000, "ns")]),
        ([pd.Timedelta(1, "ns"), pd.Timedelta(2, "ns")],
         [np.timedelta64(2, "ns"), np.timedelta64(1, "ns")]),
        pytest.param(
            [timedelta(days=200000), timedelta(days=200002)],
            [np.array(100001, dtype="timedelta64[2D]")[()],
             np.array(100000, dtype="timedelta64[2D]")[()]],
            id="duration-day-multipliers",
        ),
        pytest.param(
            [pd.Timedelta(3, "ns"), pd.Timedelta(6, "ns")],
            [np.array(2, dtype="timedelta64[3ns]")[()], np.array(1, dtype="timedelta64[3ns]")[()]],
            id="duration-nanosecond-multipliers",
        ),
        pytest.param(
            [timedelta(days=999999998), timedelta(days=999999999)],
            [np.timedelta64(999999999, "D"), np.timedelta64(999999998, "D")],
            id="wide-duration-python-limit",
        ),
        pytest.param(
            [timedelta(days=-999999999), timedelta(days=-999999998)],
            [np.timedelta64(-999999998, "D"), np.timedelta64(-999999999, "D")],
            id="wide-negative-duration-python-limit",
        ),
        pytest.param(
            [timedelta(days=200000), timedelta(days=200001)],
            [np.timedelta64(200001, "D"), np.timedelta64(200000, "D")],
            id="wide-duration-days",
        ),
        pytest.param(
            [timedelta(days=-200000, microseconds=1), timedelta(days=-200000, microseconds=2)],
            [np.timedelta64(-17279999999999998, "us"), np.timedelta64(-17279999999999999, "us")],
            id="wide-negative-duration-microseconds",
        ),
        ([pd.Timedelta(days=1), pd.Timedelta(days=2)],
         [pd.Timedelta(days=2), pd.Timedelta(days=1)]),
    ],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mixed", [False, True])
def test_timedelta_category_labels_match_equivalent_prediction_values(
    train_values, test_values, encoder, task, mixed
):
    """Equivalent durations keep known categories and sub-microsecond precision."""
    categories = pd.Index(train_values, dtype=object)
    X_train = pd.DataFrame({"signal": pd.Categorical.from_codes(np.tile([0, 1, -1], 4), categories=categories)})
    X_test = pd.DataFrame({"signal": pd.Series((test_values + [None]) * 3, dtype=object)})
    if mixed:
        X_train["constant"], X_test["constant"] = 2.25, 2.25
    before_train, before_test = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.tile([0, 1, 0], 4), np.tile([1, 0, 0], 3)
    kwargs = {"predictions": True, "n_jobs": 1, "categorical_encoder": encoder}
    if task == "classification":
        model = LazyClassifier(classifiers=[DecisionTreeClassifier], **kwargs)
        name = "DecisionTreeClassifier"
    else:
        model = LazyRegressor(regressors=[DecisionTreeRegressor], **kwargs)
        name = "DecisionTreeRegressor"
    scores, predictions = model.fit(X_train, X_test, y_train, y_test)
    assert not model.errors
    assert scores.index.tolist() == [name]
    np.testing.assert_array_equal(predictions[name], y_test)
    np.testing.assert_array_equal(model.predict(X_test, name), y_test)
    pre = model.models[name].named_steps["preprocessor"]
    encoded_train, encoded_test = pre.transform(X_train), pre.transform(X_test)
    np.testing.assert_array_equal(encoded_train[0], encoded_test[1])
    np.testing.assert_array_equal(encoded_train[1], encoded_test[0])
    assert not np.array_equal(encoded_test[0], encoded_test[1])
    assert len(pre.get_feature_names_out()) == encoded_test.shape[1]
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


@pytest.mark.parametrize("missing", [None, np.nan])
@pytest.mark.parametrize("test_dtype", ["object", "string", "category"])
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mixed", [False, True])
def test_object_missing_labels_match_extension_prediction_values(
    missing, test_dtype, encoder, task, mixed
):
    """Object-trained None remains separate from the literal missing category."""
    values = ["cedar", "missing", missing, "str:missing"]
    X_train = pd.DataFrame({"signal": pd.Series(values * 4, dtype=object)})
    X_test = pd.DataFrame({"signal": pd.Series(values * 3, dtype=test_dtype)})
    if mixed:
        X_train["constant"], X_test["constant"] = 2.25, 2.25
    target = [0, 1, 0 if missing is None else 1, 0]
    y_train, y_test = np.tile(target, 4), np.tile(target, 3)
    before_train, before_test = X_train.copy(deep=True), X_test.copy(deep=True)
    kwargs = {"predictions": True, "n_jobs": 1, "categorical_encoder": encoder}
    if task == "classification":
        model = LazyClassifier(classifiers=[DecisionTreeClassifier], **kwargs)
        name = "DecisionTreeClassifier"
    else:
        model = LazyRegressor(regressors=[DecisionTreeRegressor], **kwargs)
        name = "DecisionTreeRegressor"
    scores, predictions = model.fit(X_train, X_test, y_train, y_test)
    assert not model.errors
    assert scores.index.tolist() == [name]
    np.testing.assert_array_equal(predictions[name], y_test)
    np.testing.assert_array_equal(model.predict(X_test, name), y_test)
    pre = model.models[name].named_steps["preprocessor"]
    encoded_train, encoded_test = pre.transform(X_train), pre.transform(X_test)
    np.testing.assert_array_equal(encoded_test, encoded_train[:12])
    if missing is None:
        assert not np.array_equal(encoded_test[1], encoded_test[2])
    assert len(pre.get_feature_names_out()) == encoded_test.shape[1]
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
def test_object_none_schema_survives_arrays_refit_and_clone(encoder):
    """Fitted missing categories govern arrays, and refit resets that schema."""
    from sklearn.base import clone

    train = pd.DataFrame({"signal": pd.Series(["cedar", "missing", None, "str:missing"], dtype=object)})
    pipeline = get_categorical_encoder(encoder)
    expected = pipeline.fit_transform(train)
    for dtype in ("string", "category"):
        prediction = pd.DataFrame({"signal": pd.Series(["cedar", "missing", None, "str:missing"], dtype=dtype)})
        before = prediction.copy(deep=True)
        np.testing.assert_array_equal(pipeline.transform(prediction.to_numpy()), expected)
        pd.testing.assert_frame_equal(prediction, before)
    copied = clone(pipeline)
    assert not hasattr(copied.named_steps["to_object"], "none_columns_")
    nan_train = pd.DataFrame({"signal": pd.Series(["cedar", "missing", np.nan, "str:missing"], dtype=object)})
    refitted = pipeline.fit_transform(nan_train)
    assert not pipeline.named_steps["to_object"].none_columns_
    assert np.array_equal(refitted[1], refitted[2])
    for dtype in ("string", "category"):
        prediction = pd.DataFrame({"signal": pd.Series(["cedar", "missing", None, "str:missing"], dtype=dtype)})
        np.testing.assert_array_equal(pipeline.transform(prediction), refitted)
    pipeline.fit(train)
    np.testing.assert_array_equal(pipeline.transform(train), expected)


@pytest.mark.parametrize("test_dtype", ["string", "object", "category"])
@pytest.mark.parametrize(
    "train_dtype",
    [dtype for dtype in ("string", "str", None) if isinstance(pd.Series(["cedar"], dtype=dtype).dtype, pd.StringDtype)],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mixed", [False, True])
def test_string_missing_labels_remain_distinct_in_public_predictions(
    train_dtype, test_dtype, encoder, task, mixed
):
    """The literal missing and escape-looking text must stay separate from NA."""
    values = ["cedar", "missing", None, "str:missing"]
    X_train = pd.DataFrame({"signal": pd.Series(values * 4, dtype=train_dtype)})
    X_test = pd.DataFrame({"signal": pd.Series(values * 3, dtype=test_dtype)})
    if mixed:
        X_train["constant"], X_test["constant"] = 2.25, 2.25
    before_train, before_test = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.tile([0, 1, 0, 0], 4), np.tile([0, 1, 0, 0], 3)
    kwargs = {"predictions": True, "n_jobs": 1, "categorical_encoder": encoder}
    if task == "classification":
        model = LazyClassifier(classifiers=[DecisionTreeClassifier], **kwargs)
        name = "DecisionTreeClassifier"
    else:
        model = LazyRegressor(regressors=[DecisionTreeRegressor], **kwargs)
        name = "DecisionTreeRegressor"
    scores, predictions = model.fit(X_train, X_test, y_train, y_test)
    assert not model.errors
    assert scores.index.tolist() == [name]
    np.testing.assert_array_equal(predictions[name], y_test)
    np.testing.assert_array_equal(model.predict(X_test, name), y_test)
    pre = model.models[name].named_steps["preprocessor"]
    encoded = pre.transform(X_test)
    assert all(not np.array_equal(encoded[i], encoded[j]) for i in range(4) for j in range(i))
    assert len(pre.get_feature_names_out()) == encoded.shape[1]
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
def test_feature_names_when_base_transformer_has_no_names_api(monkeypatch, encoder):
    """A missing legacy method is a component contract, not an old package run."""
    from sklearn.preprocessing import FunctionTransformer
    monkeypatch.delattr(FunctionTransformer, "get_feature_names_out", raising=False)
    frame = pd.DataFrame({"signal": pd.Series(["cedar", "elm", "cedar"], dtype=object)})
    pipeline = get_categorical_encoder(encoder)
    pipeline.fit(frame)
    names = pipeline.get_feature_names_out()
    assert len(names) == pipeline.transform(frame).shape[1]
    assert all("signal" in name for name in names)


@pytest.mark.parametrize(
    "train_values,test_values,target,expected,known",
    [
        ([1, "1", None], [1, "1", None], [0, 1, 0], [0, 1, 0], [True, True, True]),
        ([False, "False", True, "True"], [False, "False", True, "True"],
         [0, 1, 1, 0], [0, 1, 1, 0], [True, True, True, True]),
        ([1, "1", 2, "2"], [1.0, 2.0, None], [0, 1, 0, 1], [0, 0, 0], [True, True, False]),
        ([1, "1", 2, "2"], ["1", "2", None], [0, 1, 0, 1], [1, 1, 0], [True, True, False]),
        ([1, 2], [1.0, 2.0, "1", "2"], [0, 1], [0, 1, 0, 0], [True, True, False, False]),
        (["1", "2"], ["1", "2", 1.0, 2.0], [0, 1], [0, 1, 0, 0], [True, True, False, False]),
        ([False, "False", True, "True"], [False, True, None],
         [0, 1, 1, 0], [0, 1, 0], [True, True, False]),
        ([False, True], [False, True, "False", "True"],
         [0, 1], [0, 1, 0, 0], [True, True, False, False]),
    ],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_heterogeneous_category_labels_keep_public_predictions(
    train_values, test_values, target, expected, known, encoder, mixed, task
):
    """Typed labels stay distinct and stable across independent category inference."""
    X_train = pd.DataFrame({"signal": pd.Series(train_values * 4, dtype="category")})
    X_test = pd.DataFrame({"signal": pd.Series(test_values * 3, dtype="category")})
    if mixed:
        X_train["numeric"], X_test["numeric"] = 2.25, 2.25
    before_train, before_test = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.tile(target, 4), np.tile(expected, 3)
    known_rows = np.tile(known, 3)
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
    actual = predictions[model_name].to_numpy()
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual[known_rows], y_test[known_rows])
    np.testing.assert_array_equal(estimator.predict(X_test, model_name), actual)
    preprocessor = estimator.models[model_name].named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    assert any("signal" in name for name in names)
    assert len(names) == preprocessor.transform(X_test).shape[1]
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


def test_new_transformer_keyword_respects_constructor_capabilities(monkeypatch):
    """Exercise the legacy constructor contract, not a full old sklearn runtime."""
    import lazypredict.preprocessing as preprocessing_module
    from sklearn.preprocessing import FunctionTransformer

    original_constructor = FunctionTransformer.__init__

    def limited_constructor(
        self, func=None, inverse_func=None, *, validate=False, accept_sparse=False,
        check_inverse=True, kw_args=None, inv_kw_args=None
    ):
        original_constructor(
            self,
            func=func, inverse_func=inverse_func, validate=validate,
            accept_sparse=accept_sparse, check_inverse=check_inverse,
            kw_args=kw_args, inv_kw_args=inv_kw_args,
        )

    monkeypatch.setattr(FunctionTransformer, "__init__", limited_constructor)
    pipeline = preprocessing_module.get_categorical_encoder("ordinal")
    frame = pd.DataFrame({"signal": pd.Series([1, "1", 1], dtype="category")})
    before = frame.copy(deep=True)
    encoded = pipeline.fit_transform(frame)

    assert encoded.shape == (3, 1)
    np.testing.assert_array_equal(encoded[0], encoded[2])
    assert encoded[0, 0] != encoded[1, 0]
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "train_dtype,test_dtype",
    [
        ("object", "string"), ("object", "category"),
        ("string", "object"), ("string", "category"),
        ("category", "object"), ("category", "string"),
    ],
)
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_prediction_dtype_changes_preserve_fitted_labels(train_dtype, test_dtype, encoder, mixed, task):
    """Known labels must keep their encoding when prediction dtypes change."""
    X_train = pd.DataFrame({"signal": pd.Series(["cedar", "elm"] * 6, dtype=train_dtype)})
    X_test = pd.DataFrame({"signal": pd.Series(["elm", "cedar"] * 4, dtype=test_dtype)})
    if mixed:
        X_train["constant"], X_test["constant"] = 2.25, 2.25
    before_train, before_test = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.tile([0, 1], 6), np.tile([1, 0], 4)
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
    encoded_train, encoded_test = preprocessor.transform(X_train), preprocessor.transform(X_test)
    np.testing.assert_array_equal(encoded_test[0], encoded_train[1])
    np.testing.assert_array_equal(encoded_test[1], encoded_train[0])
    assert len(preprocessor.get_feature_names_out()) == encoded_test.shape[1]
    pd.testing.assert_frame_equal(X_train, before_train)
    pd.testing.assert_frame_equal(X_test, before_test)


@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
def test_category_encoder_keeps_dataframe_labels_for_array_prediction(encoder):
    """The public encoder keeps known labels for an equivalent ndarray."""
    frame = pd.DataFrame({"signal": pd.Series(["cedar", "elm", "cedar"], dtype="category")})
    array = np.array([["elm"], ["cedar"]], dtype=object)
    before_frame, before_array = frame.copy(deep=True), array.copy()
    pipeline = get_categorical_encoder(encoder)
    pipeline.fit(frame)
    expected = pipeline.transform(frame.iloc[[1, 0]])

    np.testing.assert_array_equal(pipeline.transform(array), expected)
    assert len(pipeline.get_feature_names_out()) == expected.shape[1]
    pd.testing.assert_frame_equal(frame, before_frame)
    np.testing.assert_array_equal(array, before_array)


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

    assert normalized["value"].iloc[0] == "number:1"
    assert normalized["value"].iloc[1] == "str:1"
    assert normalized["value"].iloc[0] != normalized["value"].iloc[1]
    assert normalized["value"].iloc[2] == "missing"
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize("dtype", ["string", "str", "category"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_empty_extension_columns_remain_valid_features(dtype, mixed, encoder, task):
    """An all-missing extension column must not prevent a valid model fit."""
    X_train = pd.DataFrame({"empty": pd.Series([None] * 12, dtype=dtype)})
    X_test = pd.DataFrame({"empty": pd.Series([None] * 8, dtype=dtype)})
    if mixed:
        X_train["constant"], X_test["constant"] = 2.25, 2.25
    train_before, test_before = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.zeros(len(X_train)), np.zeros(len(X_test))
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
    assert any("empty" in name for name in preprocessor.get_feature_names_out())
    pd.testing.assert_frame_equal(X_train, train_before)
    pd.testing.assert_frame_equal(X_test, test_before)


@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_object_none_remains_distinct_from_missing_literal(encoder, task):
    """Legacy None labels must retain their existing category identity."""
    X_train = pd.DataFrame({"value": pd.Series(["cedar", None, "missing"] * 4, dtype=object)})
    X_test = pd.DataFrame({"value": pd.Series(["cedar", None, "missing"] * 3, dtype=object)})
    X_train["constant"], X_test["constant"] = 2.25, 2.25
    train_before, test_before = X_train.copy(deep=True), X_test.copy(deep=True)
    y_train, y_test = np.tile([0, 1, 0], 4), np.tile([0, 1, 0], 3)
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
