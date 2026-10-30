"""Prediction-time features must match training-time features (issue #597)."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from lazypredict.ts_preprocessing import create_lag_features, recursive_forecast


class _RecordingEstimator:
    def __init__(self):
        self.rows = []

    def predict(self, X):
        self.rows.append(np.asarray(X)[0].copy())
        return np.zeros(1)


class _IdentityScaler(StandardScaler):
    def transform(self, X, copy=None):
        return np.asarray(X, dtype=float)


def test_recursive_forecast_rebuilds_the_training_feature_row():
    y = np.random.default_rng(0).normal(size=60).cumsum()
    n_lags, n_rolling = 3, (3, 7)
    X_train, _ = create_lag_features(y, n_lags, n_rolling)

    # Features for the last training target are built from y[:-1].
    recorder = _RecordingEstimator()
    recursive_forecast(recorder, _IdentityScaler(), y[:-1], 1, n_lags, n_rolling)
    np.testing.assert_allclose(recorder.rows[0], X_train[-1])
