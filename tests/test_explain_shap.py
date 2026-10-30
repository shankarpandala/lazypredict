"""SHAP explanations through the fitted pipelines (issue #607).

shap itself is optional, so a stub module stands in for its explainers.
"""

import types

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

import lazypredict.explainability as explainability
from lazypredict.Supervised import LazyClassifier


class _StubExplainer:
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def shap_values(self, X):
        X = np.asarray(X, dtype=float)
        n_classes = len(getattr(self.model, "classes_", [None, None]))
        # old-style output: one (n_samples, n_features) array per class
        return [X * (k + 1) for k in range(n_classes)]


@pytest.fixture
def stub_shap(monkeypatch):
    fake = types.SimpleNamespace(
        TreeExplainer=_StubExplainer,
        LinearExplainer=_StubExplainer,
        PermutationExplainer=_StubExplainer,
    )
    monkeypatch.setattr(explainability, "shap", fake, raising=False)
    monkeypatch.setattr(explainability, "_SHAP_AVAILABLE", True)


def _fit(loader):
    X, y = loader(return_X_y=True, as_frame=True)
    clf = LazyClassifier(classifiers=["RandomForestClassifier", "LogisticRegression"])
    clf.fit(X.iloc[::2], X.iloc[1::2], y.iloc[::2], y.iloc[1::2])
    return clf, X.iloc[1::2], y.iloc[1::2]


def test_shap_binary_returns_positive_class(stub_shap):
    clf, X_test, y_test = _fit(load_breast_cancer)
    values = clf.explain(X_test, y_test, method="shap", max_samples=20)
    assert set(values) == {"RandomForestClassifier", "LogisticRegression"}
    for array in values.values():
        assert array.shape == (20, X_test.shape[1])
    transformed = clf.models["LogisticRegression"].named_steps["preprocessor"].transform(X_test.iloc[:20])
    np.testing.assert_allclose(values["LogisticRegression"], transformed * 2)  # class 1


def test_shap_multiclass_keeps_every_class(stub_shap):
    clf, X_test, y_test = _fit(load_iris)
    values = clf.explain(X_test, y_test, method="shap", max_samples=10)
    assert values["LogisticRegression"].shape == (10, X_test.shape[1], 3)


def test_normalise_new_style_output():
    new_style = np.ones((4, 3, 2))
    new_style[..., 1] = 7
    assert (explainability._normalise_shap_output(new_style) == 7).all()
    regression = np.zeros((4, 3))
    assert explainability._normalise_shap_output(regression).shape == (4, 3)


class _FunctionExplainer:
    """Like shap's model-agnostic explainers: one output per column of fn(X)."""

    def __init__(self, function, background, *args, **kwargs):
        self.function = function

    def shap_values(self, X):
        output = np.asarray(self.function(X), dtype=float)  # labels such as "setosa" fail, as in shap
        X = np.asarray(X, dtype=float)
        if output.ndim == 1:
            return X * output[:, None]
        return np.stack([X * output[:, [k]] for k in range(output.shape[1])], axis=-1)


class _TreeOnlyExplainer(_StubExplainer):
    def __init__(self, model, *args, **kwargs):
        if type(model).__name__.startswith("AdaBoost"):
            raise TypeError("Model type not yet supported by TreeExplainer")
        super().__init__(model)


def test_model_agnostic_classifiers_are_explained_through_probabilities(monkeypatch):
    # Issue #627: KNN, SVC, naive Bayes, ... were explained through predict(),
    # i.e. the predicted label, and AdaBoost went to TreeExplainer.
    fake = types.SimpleNamespace(
        TreeExplainer=_TreeOnlyExplainer,
        LinearExplainer=_StubExplainer,
        PermutationExplainer=_FunctionExplainer,
    )
    monkeypatch.setattr(explainability, "shap", fake, raising=False)
    monkeypatch.setattr(explainability, "_SHAP_AVAILABLE", True)
    X, y = load_iris(return_X_y=True, as_frame=True)
    y = y.map({0: "setosa", 1: "versicolor", 2: "virginica"})
    models = ["KNeighborsClassifier", "GaussianNB", "AdaBoostClassifier"]
    clf = LazyClassifier(classifiers=models)
    clf.fit(X.iloc[::2], X.iloc[1::2], y.iloc[::2], y.iloc[1::2])
    values = clf.explain(X.iloc[1::2], y.iloc[1::2], method="shap", max_samples=10)
    assert set(values) == set(models)
    for array in values.values():
        assert array.shape == (10, X.shape[1], 3)
