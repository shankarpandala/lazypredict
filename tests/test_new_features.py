"""Tests for new features: max_models, n_jobs, progress_callback, save/load, dataset warnings."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from lazypredict.Supervised import LazyClassifier, LazyRegressor


# ---------------------------------------------------------------------------
# max_models
# ---------------------------------------------------------------------------

def test_classifier_max_models():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    clf = LazyClassifier(verbose=0, ignore_warnings=True, max_models=3)
    scores, _ = clf.fit(X_train, X_test, y_train, y_test)
    assert len(scores) <= 3


def test_regressor_max_models():
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    reg = LazyRegressor(verbose=0, ignore_warnings=True, max_models=2)
    scores, _ = reg.fit(X_train, X_test, y_train, y_test)
    assert len(scores) <= 2


def test_max_models_validation():
    with pytest.raises(ValueError, match="max_models must be a positive integer"):
        LazyClassifier(max_models=0)
    with pytest.raises(ValueError, match="max_models must be a positive integer"):
        LazyClassifier(max_models=-1)


# ---------------------------------------------------------------------------
# n_jobs parameter
# ---------------------------------------------------------------------------

def test_classifier_n_jobs():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    clf = LazyClassifier(
        verbose=0, ignore_warnings=True,
        classifiers=[LogisticRegression], cv=3, n_jobs=1
    )
    scores, _ = clf.fit(X_train, X_test, y_train, y_test)
    assert len(scores) == 1


def test_n_jobs_validation():
    with pytest.raises(ValueError, match="n_jobs must be an integer"):
        LazyClassifier(n_jobs="invalid")


# ---------------------------------------------------------------------------
# progress_callback
# ---------------------------------------------------------------------------

def test_progress_callback():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )

    callback_log = []

    def my_callback(model_name, current, total, metrics):
        callback_log.append({
            "model": model_name,
            "current": current,
            "total": total,
            "has_metrics": metrics is not None,
        })

    clf = LazyClassifier(
        verbose=0, ignore_warnings=True,
        classifiers=[LogisticRegression, DecisionTreeClassifier],
        progress_callback=my_callback,
    )
    clf.fit(X_train, X_test, y_train, y_test)

    assert len(callback_log) == 2
    assert callback_log[0]["current"] == 1
    assert callback_log[1]["current"] == 2
    assert callback_log[0]["total"] == 2
    assert all(entry["has_metrics"] for entry in callback_log)


# ---------------------------------------------------------------------------
# save_models / load_models
# ---------------------------------------------------------------------------

def test_save_and_load_models():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    clf = LazyClassifier(
        verbose=0, ignore_warnings=True,
        classifiers=[LogisticRegression],
    )
    clf.fit(X_train, X_test, y_train, y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save_models(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "LogisticRegression.joblib"))

        clf2 = LazyClassifier(verbose=0, ignore_warnings=True)
        loaded = clf2.load_models(tmpdir)
        assert "LogisticRegression" in loaded
        preds = loaded["LogisticRegression"].predict(
            pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
        )
        assert len(preds) == len(y_test)


def test_save_unfitted_raises():
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="No models have been fitted"):
        clf.save_models("/tmp/dummy")


def test_load_nonexistent_dir():
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    with pytest.raises(FileNotFoundError):
        clf.load_models("/nonexistent/path")


# ---------------------------------------------------------------------------
# fit() return consistency (always tuple)
# ---------------------------------------------------------------------------

def test_fit_always_returns_tuple():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )

    # Without predictions
    clf = LazyClassifier(
        verbose=0, ignore_warnings=True, predictions=False,
        classifiers=[LogisticRegression],
    )
    result = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(result, tuple)
    assert len(result) == 2
    scores, preds_df = result
    assert isinstance(scores, pd.DataFrame)
    assert isinstance(preds_df, pd.DataFrame)

    # With predictions
    clf2 = LazyClassifier(
        verbose=0, ignore_warnings=True, predictions=True,
        classifiers=[LogisticRegression],
    )
    result2 = clf2.fit(X_train, X_test, y_train, y_test)
    assert isinstance(result2, tuple)
    scores2, preds2 = result2
    assert len(preds2) > 0


# ---------------------------------------------------------------------------
# Dataset size warnings (just ensure no crash)
# ---------------------------------------------------------------------------

def test_large_feature_warning(caplog):
    """Ensure warning is emitted for high-dimensional data."""
    import logging
    # Create high-dimensional data
    X = np.random.rand(50, 501)
    y = np.random.randint(0, 2, 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(
        verbose=0, ignore_warnings=True,
        classifiers=[LogisticRegression], max_models=1,
    )
    with caplog.at_level(logging.WARNING, logger="lazypredict"):
        clf.fit(X_train, X_test, y_train, y_test)
    assert any("High-dimensional" in msg for msg in caplog.messages)
