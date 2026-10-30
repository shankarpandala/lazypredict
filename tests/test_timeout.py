"""Per-model ``timeout`` (issue #590)."""

import signal
import threading
import time

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from lazypredict._timeout import time_limit
from lazypredict.exceptions import TimeoutException
from lazypredict.Supervised import LazyClassifier

posix_only = pytest.mark.skipif(
    not hasattr(signal, "setitimer"), reason="needs signal.setitimer (POSIX)"
)


class SlowClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, seconds=5.0):
        self.seconds = seconds

    def fit(self, X, y):
        time.sleep(self.seconds)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def _data():
    X = np.random.RandomState(0).randn(60, 3)
    y = (X[:, 0] > 0).astype(int)
    return X[:40], X[40:], y[:40], y[40:]


@posix_only
def test_slow_model_is_interrupted_and_reported():
    clf = LazyClassifier(classifiers=[SlowClassifier, "LogisticRegression"], timeout=0.5)
    start = time.perf_counter()
    scores, _ = clf.fit(*_data())
    assert time.perf_counter() - start < 3.0
    assert list(scores.index) == ["LogisticRegression"]
    assert isinstance(clf.errors["SlowClassifier"], TimeoutException)


class BrieflySlowClassifier(SlowClassifier):
    def __init__(self, seconds=0.6):
        super().__init__(seconds=seconds)


def test_timeout_is_enforced_after_the_fact_outside_the_main_thread():
    clf = LazyClassifier(classifiers=[BrieflySlowClassifier], timeout=0.2)
    worker = threading.Thread(target=clf.fit, args=_data())
    worker.start()
    worker.join()
    assert isinstance(clf.errors["BrieflySlowClassifier"], TimeoutException)
    assert "BrieflySlowClassifier" not in clf.models


@posix_only
def test_time_limit_leaves_existing_timers_alone():
    previous = signal.signal(signal.SIGALRM, lambda *args: None)
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        with time_limit(0.1) as enforced:
            assert not enforced
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@posix_only
def test_time_limit_restores_handler_and_disarms_timer():
    previous = signal.getsignal(signal.SIGALRM)
    with time_limit(5) as enforced:
        assert enforced
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert signal.getsignal(signal.SIGALRM) is previous


def test_no_limit():
    with time_limit(None) as enforced:
        assert not enforced
