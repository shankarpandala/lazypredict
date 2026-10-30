"""Seasonal period auto-detection (issue #600)."""

import numpy as np
import pytest

from lazypredict.ts_preprocessing import detect_seasonal_period


def _rng(seed):
    return np.random.default_rng(seed)


SEASONAL = {
    "trend_multiplicative_12": (
        12, lambda r: (100 + 2 * np.arange(144)) * (1 + 0.15 * np.sin(2 * np.pi * np.arange(144) / 12))
        + r.normal(0, 5, 144),
    ),
    "quarterly_4": (4, lambda r: 10 + 0.1 * np.arange(80) + 2 * np.sin(np.pi * np.arange(80) / 2) + r.normal(0, 1, 80)),
    "weekly_7": (7, lambda r: 50 + np.where(np.arange(365) % 7 >= 5, 8, 0) + r.normal(0, 3, 365)),
    "random_walk_plus_12": (
        12, lambda r: np.cumsum(r.normal(0, 0.3, 200)) + 3 * np.sin(2 * np.pi * np.arange(200) / 12)
        + r.normal(0, 0.5, 200),
    ),
}

NON_SEASONAL = {
    "white_noise": lambda r: r.normal(size=200),
    "trend_plus_noise": lambda r: 0.05 * np.arange(200) + r.normal(size=200),
    "random_walk": lambda r: np.cumsum(r.normal(size=200)),
    "ar1": lambda r: _ar1(r, 300, 0.8),
}


def _ar1(r, n, phi):
    e = r.normal(size=n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + e[i]
    return y


@pytest.mark.parametrize("name", list(SEASONAL))
def test_detects_true_period(name):
    period, make = SEASONAL[name]
    detected = [detect_seasonal_period(make(_rng(seed))) for seed in range(40)]
    assert detected.count(period) >= 36, detected


@pytest.mark.parametrize("name", list(NON_SEASONAL))
def test_no_period_for_non_seasonal_series(name):
    detected = [detect_seasonal_period(NON_SEASONAL[name](_rng(seed))) for seed in range(40)]
    assert sum(d is not None for d in detected) <= 2, detected


@pytest.mark.parametrize("y", [np.ones(50), np.arange(5.0), np.array([1.0, np.nan] * 20)])
def test_degenerate_inputs(y):
    assert detect_seasonal_period(y) is None
