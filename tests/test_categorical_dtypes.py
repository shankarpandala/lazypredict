"""pandas ``category``/``string``/``str`` columns are used as features (issue #582)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from lazypredict.preprocessing import build_preprocessor, categories_as_strings
from lazypredict.Supervised import LazyClassifier, LazyRegressor

N = 240


@pytest.fixture
def labels():
    return np.random.RandomState(0).choice(["a", "b", "c"], N)


def _frame(labels, dtype):
    rng = np.random.RandomState(1)
    if dtype == "category":
        column = pd.Categorical(labels)
    elif dtype == "object":
        column = labels.astype(object)
    else:
        column = pd.array(labels, dtype=dtype)
    return pd.DataFrame({"num": rng.randn(len(labels)), "cat": column})


@pytest.mark.parametrize("dtype", ["category", "string", "str", "object"])
@pytest.mark.parametrize("encoder", ["onehot", "ordinal"])
def test_categorical_dtypes_are_encoded(labels, dtype, encoder):
    X = _frame(labels, dtype)
    y = (labels == "a").astype(int)
    clf = LazyClassifier(classifiers=["LogisticRegression"], categorical_encoder=encoder)
    with warnings.catch_warnings():
        # pandas 3 deprecates selecting ``str`` columns through ``include="object"``
        warnings.filterwarnings("error", message=".*select_dtypes.*")
        scores, _ = clf.fit(X[:180], X[180:], y[:180], y[180:])
    assert scores.loc["LogisticRegression", "Accuracy"] == 1.0
    transformed = clf.models["LogisticRegression"].named_steps["preprocessor"]
    assert "cat" not in transformed.transformers_[-1][2]  # not left in the remainder


def test_train_and_test_dtypes_may_differ(labels):
    rng = np.random.RandomState(2)
    train_labels = np.where(rng.rand(180) < 0.1, None, labels[:180])
    test_labels = np.where(rng.rand(60) < 0.1, None, labels[180:]).astype(object)
    test_labels[0] = "unseen"
    X_train = pd.DataFrame({"num": rng.randn(180), "cat": pd.Categorical(train_labels)})
    X_test = pd.DataFrame({"num": rng.randn(60), "cat": pd.array(test_labels, dtype="string")})
    y = (labels == "a").astype(int)
    clf = LazyClassifier(classifiers=["LogisticRegression"])
    scores, _ = clf.fit(X_train, X_test, y[:180], y[180:])
    assert not clf.errors
    assert scores.loc["LogisticRegression", "Accuracy"] > 0.8


def test_numeric_categories_mixed_objects_and_empty_columns(labels):
    rng = np.random.RandomState(3)
    X = pd.DataFrame({
        "num": rng.randn(N),
        "cat_int": pd.Categorical(np.where(rng.rand(N) < 0.1, np.nan, (labels == "a").astype(float))),
        "mixed": pd.Series([1 if label == "a" else label for label in labels], dtype=object),
        "empty": pd.Series([None] * N, dtype="string"),
    })
    y = (labels == "a") * 3.0 + rng.randn(N) * 0.1
    reg = LazyRegressor(regressors=["LinearRegression"])
    scores, _ = reg.fit(X[:180], X[180:], y[:180], y[180:])
    assert not reg.errors
    assert scores.loc["LinearRegression", "R-Squared"] > 0.9


def test_categories_as_strings_normalises_missing_values():
    frame = pd.DataFrame({
        "obj": pd.Series(["x", None, np.nan], dtype=object),
        "string": pd.array(["x", pd.NA, "y"], dtype="string"),
        "cat": pd.Categorical([1, None, 2]),
    })
    out = categories_as_strings(frame)
    assert out.dtype == object
    assert out[0].tolist() == ["x", "x", "1"]
    assert all(isinstance(v, float) and np.isnan(v) for v in out[1])
    assert out[2, 1:].tolist() == ["y", "2"]


def test_feature_names_out(labels):
    X = _frame(labels, "category")
    pre = build_preprocessor(X, "onehot").fit(X)
    names = list(pre.get_feature_names_out())
    assert "numeric__num" in names
    assert any(name.startswith("categorical_low__cat_") for name in names)


def test_equal_numeric_codes_get_equal_labels():
    # Issue #620: integer codes turn into floats when one value is missing.
    values = pd.DataFrame(
        {"code": pd.Series([1, 2.0, np.nan, np.int64(3), True, 2.5, "x"], dtype=object)}
    )
    result = categories_as_strings(values)[:, 0]
    assert [result[i] for i in (0, 1, 3, 4, 5, 6)] == ["1", "2", "3", "True", "2.5", "x"]
    assert pd.isna(result[2])


@pytest.mark.parametrize("dtype", ["object", "category"])
def test_integer_codes_match_float_codes_in_the_test_split(dtype):
    rng = np.random.RandomState(0)
    store = rng.randint(1, 6, N)
    y = (store >= 3).astype(int)
    X = pd.DataFrame({"store": store, "noise": rng.randn(N)})
    train, test = X.iloc[:180].copy(), X.iloc[180:].copy()
    test["store"] = test["store"].astype(float)
    test.loc[test.index[0], "store"] = np.nan
    train["store"] = train["store"].astype(dtype)
    test["store"] = test["store"].astype(dtype)
    clf = LazyClassifier(classifiers=["LogisticRegression"])
    scores, _ = clf.fit(train, test, y[:180], y[180:])
    assert scores.loc["LogisticRegression", "Accuracy"] > 0.95
