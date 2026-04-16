"""
Time Series Forecasting Example
================================

Benchmark 20+ forecasting models on a synthetic seasonal time series.

Requires: pip install lazypredict[timeseries]
"""

import numpy as np

from lazypredict.TimeSeriesForecasting import LazyForecaster

# 1. Generate a synthetic seasonal time series
np.random.seed(42)
t = np.arange(200)
y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 200)

y_train, y_test = y[:180], y[180:]

# 2. Fit all forecasting models
fcst = LazyForecaster(verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)

print("=== Time Series Forecasting Results ===")
print(scores)

# 3. (Optional) With exogenous variables
X_train = np.column_stack([np.sin(t[:180]), np.cos(t[:180])])
X_test = np.column_stack([np.sin(t[180:]), np.cos(t[180:])])

scores_exog, predictions_exog = fcst.fit(y_train, y_test, X_train, X_test)

print("\n=== With Exogenous Variables ===")
print(scores_exog)
