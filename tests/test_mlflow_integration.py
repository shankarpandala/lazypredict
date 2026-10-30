"""MLflow integration against a recording fake ``mlflow`` module (issue #608)."""

import types

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

import lazypredict._base as base
import lazypredict.TimeSeriesForecasting as tsf
from lazypredict.integrations import mlflow as integration
from lazypredict.Supervised import LazyClassifier
from lazypredict.TimeSeriesForecasting import LazyForecaster


class FakeMlflow:
    """Records calls; refuses a non-nested run while another run is active."""

    def __init__(self, model_api="mlflow3"):
        self.calls = []
        self.active = []
        self.metrics = {}
        self.models = types.SimpleNamespace(infer_signature=lambda X, y: "signature")
        if model_api == "mlflow3":
            def log_model(sk_model, name=None, signature=None, registered_model_name=None, **kw):
                self.calls.append(("log_model", name, registered_model_name))
        else:
            def log_model(sk_model, artifact_path, signature=None, registered_model_name=None, **kw):
                self.calls.append(("log_model", artifact_path, registered_model_name))
        self.sklearn = types.SimpleNamespace(log_model=log_model)

    def start_run(self, run_name=None, nested=False):
        if self.active and not nested:
            raise RuntimeError("Run is already active. To start a nested run, call start_run with nested=True")
        self.active.append(run_name)
        self.calls.append(("start_run", run_name, nested))
        return run_name

    def end_run(self):
        self.active.pop()

    def log_param(self, key, value):
        self.calls.append(("log_param", key, value))

    def log_metric(self, key, value):
        assert np.isfinite(value), (key, value)
        self.metrics[key] = value

    def autolog(self, *args, **kwargs):  # must never be called
        raise AssertionError("mlflow.autolog() must not be called")

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setattr(integration, "mlflow", fake, raising=False)
    monkeypatch.setattr(integration, "MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(base, "_mlflow", fake)
    monkeypatch.setattr(base, "MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(tsf, "_mlflow", fake)
    monkeypatch.setattr(tsf, "MLFLOW_AVAILABLE", True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
    return fake


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return X[:400], X[400:], y[:400], y[400:]


def test_supervised_runs_nest_inside_an_active_user_run(fake_mlflow):
    clf = LazyClassifier(classifiers=["LogisticRegression", "DecisionTreeClassifier"])
    assert clf.mlflow_enabled
    fake_mlflow.start_run(run_name="user-run")  # the user's own run
    scores, _ = clf.fit(*_split())
    assert not clf.errors and len(scores) == 2
    runs = [c for c in fake_mlflow.calls if c[0] == "start_run" and c[1] != "user-run"]
    assert runs and all(nested for _, _, nested in runs)
    assert fake_mlflow.active == ["user-run"]  # every model run was closed


def test_models_are_logged_but_not_registered(fake_mlflow):
    LazyClassifier(classifiers=["LogisticRegression"]).fit(*_split())
    logged = [c for c in fake_mlflow.calls if c[0] == "log_model"]
    assert logged == [("log_model", "LogisticRegression_model", None)]
    assert "accuracy" in fake_mlflow.metrics


def test_mlflow2_signature_uses_artifact_path(monkeypatch):
    fake = FakeMlflow(model_api="mlflow2")
    integration.log_sklearn_model(fake, object(), "Model_model", signature=None)
    assert fake.calls == [("log_model", "Model_model", None)]


def test_forecaster_runs_are_nested(fake_mlflow):
    y = np.sin(np.arange(80) / 3.0) + np.arange(80) * 0.05
    fake_mlflow.start_run(run_name="user-run")
    scores, _ = LazyForecaster(forecasters=["Naive", "Ridge_TS"]).fit(y[:70], y[70:])
    assert len(scores) == 2
    runs = [c for c in fake_mlflow.calls if c[0] == "start_run" and c[1] != "user-run"]
    assert runs and all(nested for _, _, nested in runs)
