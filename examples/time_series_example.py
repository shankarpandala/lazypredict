# time_series_example.py
import numpy as np
import pandas as pd
from lazypredict.estimators.time_series import LazyTimeSeriesForecaster
from lazypredict.metrics.time_series_metrics import TimeSeriesMetrics
from lazypredict.utils.logger import Logger

# Generate synthetic time series data
np.random.seed(42)
n_points = 100
time = np.arange(n_points)
y = 0.5 * time + np.sin(time) + np.random.normal(scale=5, size=n_points)

# Train-test split
train_size = int(n_points * 0.8)
y_train, y_test = y[:train_size], y[train_size:]

# Logger setup
logger = Logger.configure_logger("time_series_example")

# Model Training and Evaluation
model = LazyTimeSeriesForecaster()
model.fit(time[:train_size], y_train)
predictions = model.predict(steps=len(y_test))

# Compute Metrics
metrics_calculator = TimeSeriesMetrics()
metrics = metrics_calculator.compute(y_test, predictions)
print("Metrics:", metrics)
