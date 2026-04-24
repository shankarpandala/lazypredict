# Time Series Forecasting

`LazyForecaster` benchmarks 20+ statistical, ML, deep-learning, and foundation
models on your time series in a single call.

---

## Quick Start

```python
import numpy as np
from lazypredict.TimeSeriesForecasting import LazyForecaster

# Sample data: trend + seasonal + noise
np.random.seed(42)
t = np.arange(200)
y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 200)

y_train, y_test = y[:180], y[180:]

fcst = LazyForecaster(verbose=1, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)
print(scores)
```

---

## Available Forecasting Models

### Baseline Models (always available)

| Name | Class | Description |
|------|-------|-------------|
| `Naive` | `NaiveForecaster` | Repeats the last observed value |
| `SeasonalNaive` | `SeasonalNaiveForecaster` | Repeats the last seasonal cycle |

### Statistical Models (require `pip install lazypredict[timeseries]`)

| Name | Class | Description |
|------|-------|-------------|
| `SimpleExpSmoothing` | `SimpleExpSmoothingForecaster` | Simple exponential smoothing |
| `Holt` | `HoltForecaster` | Double exponential smoothing (trend) |
| `HoltWinters_Add` | `HoltWintersForecaster` | Holt-Winters additive seasonality |
| `HoltWinters_Mul` | `HoltWintersForecaster` | Holt-Winters multiplicative seasonality |
| `Theta` | `ThetaForecaster` | Theta decomposition method |
| `SARIMAX` | `SARIMAXForecaster` | Seasonal ARIMA with exogenous regressors |
| `AutoARIMA` | `AutoARIMAForecaster` | Automatic ARIMA order selection |

### Machine Learning Models (always available)

| Name | Class | Description |
|------|-------|-------------|
| `LinearRegression_TS` | `MLForecaster` | Linear regression with lag features |
| `Ridge_TS` | `MLForecaster` | Ridge regression with lag features |
| `Lasso_TS` | `MLForecaster` | Lasso regression with lag features |
| `ElasticNet_TS` | `MLForecaster` | ElasticNet with lag features |
| `RandomForest_TS` | `MLForecaster` | Random forest with lag features |
| `ExtraTrees_TS` | `MLForecaster` | Extra trees with lag features |
| `GradientBoosting_TS` | `MLForecaster` | Gradient boosting with lag features |
| `AdaBoost_TS` | `MLForecaster` | AdaBoost with lag features |
| `Bagging_TS` | `MLForecaster` | Bagging regressor with lag features |
| `KNeighbors_TS` | `MLForecaster` | k-NN regression with lag features |
| `SVR_TS` | `MLForecaster` | Support vector regression |
| `DecisionTree_TS` | `MLForecaster` | Decision tree with lag features |
| `XGBoost_TS` | `MLForecaster` | XGBoost (requires `lazypredict[boost]`) |
| `LightGBM_TS` | `MLForecaster` | LightGBM (requires `lazypredict[boost]`) |
| `CatBoost_TS` | `MLForecaster` | CatBoost (requires `lazypredict[boost]`) |

### Deep Learning Models (require `pip install lazypredict[deeplearning]`)

| Name | Class | Description |
|------|-------|-------------|
| `LSTM_TS` | `_TorchRNNForecaster` | Long Short-Term Memory network |
| `GRU_TS` | `_TorchRNNForecaster` | Gated Recurrent Unit network |

### Foundation Models (require `pip install lazypredict[foundation]`)

| Name | Class | Description |
|------|-------|-------------|
| `TimesFM` | `TimesFMForecaster` | Google TimesFM 2.5 pretrained model |

!!! note "Python version restriction"
    TimesFM requires Python 3.10 or 3.11.

---

## Parameters Reference

```python
LazyForecaster(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None,
    predictions=False,
    random_state=42,
    forecasters="all",       # list of model names, or "all"
    cv=None,                 # int >= 2 for cross-validation
    timeout=None,            # seconds per model
    n_lags=10,               # lag features for ML models
    n_rolling=(3, 7),        # rolling window sizes for ML models
    seasonal_period=None,    # override auto-detection (None = auto)
    sort_by="RMSE",          # metric to sort results by
    n_jobs=-1,
    max_models=None,
    use_gpu=False,
    foundation_model_path=None,  # local path for offline TimesFM
    # Tuning
    tune=False,
    tune_top_k=5,
    tune_trials=30,
    tune_timeout=None,
    tune_metric="RMSE",
    tune_seasonal=False,
    # Horizon strategy
    horizon_strategy="recursive",  # "recursive", "direct", "multi_output"
)
```

---

## Evaluating Results

The `scores` DataFrame returned by `fit()` contains:

| Column | Description |
|--------|-------------|
| `MAE` | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |
| `MAPE` | Mean Absolute Percentage Error (%) |
| `SMAPE` | Symmetric MAPE (%) |
| `MASE` | Mean Absolute Scaled Error (< 1 beats naive) |
| `R-Squared` | R² coefficient of determination |
| `Time Taken` | Seconds to fit and predict |

When `cv` is set, additional columns appear:

`MAE CV Mean`, `MAE CV Std`, `RMSE CV Mean`, `RMSE CV Std`, etc.

---

## Exogenous Variables

Pass exogenous features to models that support them (SARIMAX, AutoARIMA, ML models):

```python
X_train = np.column_stack([np.sin(t[:180]), np.cos(t[:180])])
X_test  = np.column_stack([np.sin(t[180:]), np.cos(t[180:])])

fcst = LazyForecaster(verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test, X_train, X_test)
```

---

## Auto Seasonal Period Detection

By default the seasonal period is auto-detected via autocorrelation. Override manually:

```python
fcst = LazyForecaster(seasonal_period=12)   # monthly data with yearly cycle
```

Set `seasonal_period=1` to disable seasonal models.

---

## Cross-Validation

Use `TimeSeriesSplit` expanding-window cross-validation:

```python
fcst = LazyForecaster(cv=5, verbose=0)
scores, predictions = fcst.fit(y_train, y_test)
# scores includes "MAE CV Mean", "RMSE CV Mean", etc.
```

---

## Selecting Specific Models

Run only a subset of models by name:

```python
fcst = LazyForecaster(
    forecasters=["Naive", "Holt", "AutoARIMA", "Ridge_TS", "LSTM_TS"],
    verbose=0,
)
scores, predictions = fcst.fit(y_train, y_test)
```

---

## Custom Metric

```python
def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

fcst = LazyForecaster(custom_metric=median_absolute_error, verbose=0)
scores, predictions = fcst.fit(y_train, y_test)
# scores includes a "median_absolute_error" column
```

---

## Horizon Strategies for ML Models

Control how ML models forecast multiple steps ahead:

```python
# Recursive (default): single model, autoregressively forecasts step by step
fcst = LazyForecaster(horizon_strategy="recursive")

# Direct: one model per horizon step (more accurate for longer horizons)
fcst = LazyForecaster(horizon_strategy="direct")

# Multi-output: MultiOutputRegressor wrapper
fcst = LazyForecaster(horizon_strategy="multi_output")
```

---

## Ensemble Predictions

After calling `fit()` with `predictions=True`, combine model predictions:

```python
fcst = LazyForecaster(predictions=True, verbose=0)
scores, pred_df = fcst.fit(y_train, y_test)

# Simple average of all model predictions
avg = fcst.ensemble(method="simple_average")

# Inverse-error weighted average (lower error = higher weight)
weighted = fcst.ensemble(method="weighted_average", y_true=y_test)

# Stacking (linear meta-learner on top)
stacked = fcst.ensemble(method="stacking", y_true=y_test)
```

---

## Saving and Loading Models

```python
# Save fitted models to disk
fcst.save_models("./my_forecasters")

# Load in a new session
fcst2 = LazyForecaster()
fcst2.load_models("./my_forecasters")

# Forecast with loaded models
forecasts = fcst2.predict(y_history=y_train, horizon=20)
```

---

## Visualization

Requires `pip install lazypredict[viz]`:

```python
fcst = LazyForecaster(predictions=True, verbose=0)
scores, pred_df = fcst.fit(y_train, y_test)

# Plot forecasts vs actuals for all models
fcst.plot_results(y_train=y_train, y_test=y_test, plot_type="forecast")

# Bar chart comparison of model errors
fcst.plot_results(plot_type="comparison", metric="RMSE", top_k=10)

# Residual diagnostics for a single model
fcst.plot_results(plot_type="residuals", model_name="HoltWinters_Add")

# Error distribution violin plot
fcst.plot_results(plot_type="errors")

# Metrics heatmap across all models
fcst.plot_results(plot_type="heatmap")
```

---

## Offline / Air-Gapped TimesFM

If you cannot download the TimesFM weights at runtime:

```python
# Pre-download once (requires internet)
from huggingface_hub import snapshot_download
snapshot_download("google/timesfm-2.5-200m-pytorch", local_dir="./timesfm-local")

# Use locally in air-gapped environments
fcst = LazyForecaster(foundation_model_path="./timesfm-local")
scores, predictions = fcst.fit(y_train, y_test)
```

---

## GPU Acceleration

```python
fcst = LazyForecaster(use_gpu=True, verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)
```

GPU-accelerated forecasting models:

- XGBoost_TS, LightGBM_TS, CatBoost_TS
- LSTM_TS, GRU_TS (PyTorch CUDA)
- TimesFM (PyTorch CUDA)

!!! note
    CUDA must be available and PyTorch must be installed with CUDA support.
    All models fall back to CPU automatically if CUDA is not detected.

---

## MLflow Tracking

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

fcst = LazyForecaster(verbose=0)
scores, predictions = fcst.fit(y_train, y_test)
# MAE, RMSE, MAPE, SMAPE, MASE, Time Taken logged per model
```
