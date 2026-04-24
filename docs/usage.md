# Usage Guide

## Importing

```python
import lazypredict
```

---

## Classification

Benchmark all available classifiers on a dataset in one call:

```python
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.5, random_state=123
)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
```

The returned `models` DataFrame is sorted by Accuracy and includes:
Accuracy, Balanced Accuracy, ROC AUC, F1 Score, and Time Taken.

---

## Regression

```python
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import numpy as np

diabetes = load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
```

The `models` DataFrame includes: R-Squared, Adjusted R-Squared, RMSE, and Time Taken.

!!! warning "Deprecated class names"
    The old `Regression` and `Classification` class names are deprecated.
    Use `LazyRegressor` and `LazyClassifier` instead.

---

## Time Series Forecasting

`LazyForecaster` benchmarks 20+ statistical, machine-learning, deep-learning,
and foundation models with a single call:

```python
import numpy as np
from lazypredict.TimeSeriesForecasting import LazyForecaster

np.random.seed(42)
t = np.arange(200)
y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 200)

y_train, y_test = y[:180], y[180:]

fcst = LazyForecaster(verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)
print(scores)
```

See the [Time Series Forecasting](forecasting.md) page for the full guide.

---

## Common Parameters

All `LazyClassifier`, `LazyRegressor`, and `LazyForecaster` share these options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | int | 0 | Verbosity level (0 = silent) |
| `ignore_warnings` | bool | True | Suppress per-model errors and warnings |
| `custom_metric` | callable | None | Extra metric `f(y_true, y_pred)` |
| `predictions` | bool | False | Return per-model predictions DataFrame |
| `random_state` | int | 42 | Random seed |
| `cv` | int or None | None | Cross-validation folds (≥ 2) |
| `timeout` | float or None | None | Max seconds per model |
| `n_jobs` | int | -1 | Parallel jobs (-1 = all CPUs) |
| `max_models` | int or None | None | Limit number of models trained |
| `use_gpu` | bool | False | Enable GPU acceleration |

Classifier/Regressor-only:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `categorical_encoder` | str | `'onehot'` | Encoder: `'onehot'`, `'ordinal'`, `'target'`, `'binary'` |
| `tune` | bool | False | Tune top-k models after benchmarking |
| `tune_top_k` | int | 5 | Number of models to tune |
| `tune_trials` | int | 50 | Optuna trials per model |
| `tune_backend` | str | `'optuna'` | Tuning backend: `'optuna'`, `'sklearn'`, `'flaml'` |

---

## Getting Trained Model Objects

Access the fitted model pipelines after calling `fit()`:

```python
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Access all fitted model objects
model_dict = reg.provide_models(X_train, X_test, y_train, y_test)

# Use a specific model
rf = model_dict["RandomForestRegressor"]
preds = rf.predict(X_test)
```

---

## GPU Acceleration

Enable GPU with `use_gpu=True`:

```python
clf = LazyClassifier(use_gpu=True, verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

reg = LazyRegressor(use_gpu=True, verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

fcst = LazyForecaster(use_gpu=True, verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)
```

Supported GPU backends:

- **XGBoost** (`device="cuda"`)
- **LightGBM** (`device="gpu"`)
- **CatBoost** (`task_type="GPU"`)
- **cuML (RAPIDS)** — GPU-native sklearn replacements (auto-discovered)
- **LSTM / GRU** — PyTorch CUDA
- **TimesFM** — PyTorch CUDA

Falls back to CPU automatically if no CUDA GPU is available.

---

## Command-Line Interface

Lazy Predict ships a CLI for quick benchmarking from a CSV file:

```console
# Classification
lazypredict --task classification --input data.csv --target label

# Regression
lazypredict --task regression --input data.csv --target price --test-size 0.2
```

Options:

| Flag | Description |
|------|-------------|
| `--task` | `classification` or `regression` |
| `--input` | Path to a CSV file |
| `--target` | Column name for the target variable |
| `--test-size` | Fraction for test split (default: 0.2) |
| `--random-state` | Random seed (default: 42) |
| `--version` | Show version |
| `--help` | Show help |

---

## Logging Configuration

Configure the `lazypredict` logger to control output:

```python
import logging

# Show info-level messages from all models
logging.getLogger("lazypredict").setLevel(logging.INFO)

# Suppress everything
logging.getLogger("lazypredict").setLevel(logging.ERROR)

# Log to file
handler = logging.FileHandler("lazypredict.log")
logging.getLogger("lazypredict").addHandler(handler)
```

---

## MLflow Integration

Enable automatic experiment tracking by setting `MLFLOW_TRACKING_URI` before running:

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
# Metrics and model artifacts are automatically logged to MLflow
```

View results in the MLflow UI:

```console
mlflow ui
```

For Databricks, MLflow tracking is configured automatically from the environment.
