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

    def log_params(self, params):
        self.calls.append(("log_params", dict(params)))

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


# -- Tracking never changes the benchmark (issue #626) ---------------------------

MODELS = ["LogisticRegression", "DecisionTreeClassifier"]


def test_unreachable_tracking_server_does_not_fail_models(fake_mlflow, monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionError("tracking server unreachable")

    monkeypatch.setattr(fake_mlflow, "start_run", refuse)
    clf = LazyClassifier(classifiers=MODELS)
    scores, _ = clf.fit(*_split())
    assert sorted(scores.index) == sorted(MODELS)
    assert clf.errors == {}


def test_failed_metric_logging_keeps_the_model_and_ends_its_run(fake_mlflow, monkeypatch):
    def refuse(key, value):
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(fake_mlflow, "log_metric", refuse)
    clf = LazyClassifier(classifiers=MODELS)
    scores, _ = clf.fit(*_split())
    assert sorted(scores.index) == sorted(MODELS) and clf.errors == {}
    assert fake_mlflow.active == []
    # Tracking was turned off after the failure: only the first model got a run.
    assert [c[1] for c in fake_mlflow.calls if c[0] == "start_run"] == [f"LazyClassifier-{MODELS[0]}"]


def test_lambda_custom_metric_is_kept_and_logged_under_a_valid_key(fake_mlflow, monkeypatch):
    def log_metric(key, value):
        if any(ch in key for ch in "<>"):
            raise ValueError(f"Invalid metric name: {key!r}")
        fake_mlflow.metrics[key] = value

    monkeypatch.setattr(fake_mlflow, "log_metric", log_metric)
    clf = LazyClassifier(classifiers=MODELS, custom_metric=lambda y, p: float(np.mean(y == p)))
    scores, _ = clf.fit(*_split())
    assert scores["<lambda>"].notna().all()
    assert "custom_metric" in fake_mlflow.metrics


def test_failing_progress_callback_does_not_end_the_users_run(fake_mlflow):
    def callback(name, done, total, metrics):
        raise KeyError("Accuracy")

    fake_mlflow.start_run(run_name="user run")
    clf = LazyClassifier(classifiers=MODELS, progress_callback=callback)
    with pytest.raises(KeyError):
        clf.fit(*_split())
    assert fake_mlflow.active == ["user run"]


def test_forecaster_keeps_models_when_tracking_fails(fake_mlflow, monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionError("tracking server unreachable")

    monkeypatch.setattr(fake_mlflow, "start_run", refuse)
    y = np.sin(np.arange(120) / 3.0) + np.arange(120) * 0.05
    forecaster = LazyForecaster(forecasters=["Naive", "Ridge_TS"])
    scores, _ = forecaster.fit(y[:100], y[100:])
    assert sorted(scores.index) == ["Naive", "Ridge_TS"]
    assert forecaster.errors == {}


def test_estimator_hyperparameters_are_logged(fake_mlflow):
    LazyClassifier(classifiers=["LogisticRegression"]).fit(*_split())
    logged = [c[1] for c in fake_mlflow.calls if c[0] == "log_params"]
    assert logged and logged[0]["C"] == "1.0"
