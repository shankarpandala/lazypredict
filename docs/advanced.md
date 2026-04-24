# Advanced Features

This page covers advanced Lazy Predict capabilities beyond the basic benchmarking workflow.

---

## Hyperparameter Tuning

After benchmarking, automatically tune the top-k models using Optuna, scikit-learn, or FLAML.

### Installation

```console
pip install lazypredict[tune]    # Optuna (default)
pip install lazypredict[flaml]   # FLAML alternative
```

### Supervised Models

```python
from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(
    verbose=1,
    tune=True,
    tune_top_k=5,          # tune the 5 best models
    tune_trials=50,         # 50 Optuna trials per model
    tune_timeout=120,       # max 2 minutes per model
    tune_backend="optuna",  # "optuna", "sklearn", or "flaml"
)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

### Time Series

```python
from lazypredict.TimeSeriesForecasting import LazyForecaster

fcst = LazyForecaster(
    tune=True,
    tune_top_k=3,
    tune_trials=30,
    tune_metric="RMSE",     # "RMSE", "MAE", "MAPE", "SMAPE", "MASE"
    tune_seasonal=False,    # also search over seasonal_period
)
scores, predictions = fcst.fit(y_train, y_test)

# Access tuning results
print(fcst.tuned_scores_)
```

### Using the Tuning API Directly

```python
from lazypredict.tuning import tune_model_optuna, tune_model_sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Build a fitted pipeline first
pipeline = Pipeline([("model", RandomForestClassifier())])
pipeline.fit(X_train, y_train)

# Tune with Optuna
best_params = tune_model_optuna(
    model_class=RandomForestClassifier,
    X_train=X_train,
    y_train=y_train,
    scoring="balanced_accuracy",
    n_trials=100,
)
```

---

## Explainability

Compute feature importance for all benchmarked models.

### Installation

```console
pip install lazypredict[explain]    # SHAP
pip install lazypredict[interpret]  # InterpretML (EBM)
```

### Permutation Importance (no extra dependencies)

```python
from lazypredict.Supervised import LazyRegressor
from lazypredict.explainability import explain_permutation

reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Get fitted pipeline objects
model_dict = reg.provide_models(X_train, X_test, y_train, y_test)

# Compute permutation importance for all models
importance_df = explain_permutation(
    models=model_dict,
    X_test=X_test,
    y_test=y_test,
    n_repeats=10,
    random_state=42,
)
print(importance_df)
```

### SHAP Values

```python
from lazypredict.explainability import explain_shap

# Get SHAP values for a single model
shap_df = explain_shap(
    model=model_dict["RandomForestRegressor"],
    X_test=X_test,
    feature_names=feature_names,
)
print(shap_df)
```

### InterpretML (EBM)

When `interpret` is installed, `ExplainableBoostingClassifier` and
`ExplainableBoostingRegressor` are automatically included in benchmarking.
These are inherently interpretable glass-box models.

---

## Time Series Visualization

Requires `pip install lazypredict[viz]`.

### Via `LazyForecaster.plot_results()`

```python
fcst = LazyForecaster(predictions=True, verbose=0)
scores, pred_df = fcst.fit(y_train, y_test)

# Overlay all model forecasts on actual series
fig = fcst.plot_results(plot_type="forecast")

# Bar chart ranking models by a metric
fig = fcst.plot_results(plot_type="comparison", metric="MAE", top_k=10)

# Residual diagnostics (ACF, histogram, Q-Q)
fig = fcst.plot_results(plot_type="residuals", model_name="Ridge_TS")

# Violin plot of absolute errors per model
fig = fcst.plot_results(plot_type="errors")

# Heatmap of all metrics × all models
fig = fcst.plot_results(plot_type="heatmap")
```

### Direct Plotting Functions

```python
from lazypredict.ts_visualization import (
    plot_forecast,
    plot_model_comparison,
    plot_residuals,
    plot_error_distribution,
    plot_metrics_heatmap,
)

# Forecast overlay
fig = plot_forecast(y_train, y_test, predictions={"Ridge_TS": y_pred})

# Model comparison bar chart
fig = plot_model_comparison(scores, metric="RMSE", top_k=10)

# Residuals for one model
fig = plot_residuals(y_test, y_pred, model_name="Ridge_TS", seasonal_period=12)

# Error distribution
fig = plot_error_distribution(y_test, predictions_dict)

# Metrics heatmap
fig = plot_metrics_heatmap(scores)
```

---

## Ensemble Methods

Combine predictions from multiple models after benchmarking.

### Via `LazyForecaster.ensemble()`

```python
fcst = LazyForecaster(predictions=True, verbose=0)
scores, pred_df = fcst.fit(y_train, y_test)

# Simple average
avg_pred = fcst.ensemble(method="simple_average")

# Inverse-error weighted (lower RMSE = higher weight)
weighted_pred = fcst.ensemble(method="weighted_average", y_true=y_test)

# Stacking (linear regression meta-model)
stacked_pred = fcst.ensemble(method="stacking", y_true=y_test)
```

### Direct Ensemble Functions

```python
from lazypredict.ensemble import (
    ensemble_simple_average,
    ensemble_weighted_average,
    ensemble_stacking,
)

predictions = {
    "Holt": pred_holt,
    "Ridge_TS": pred_ridge,
    "LSTM_TS": pred_lstm,
}

# Simple average
avg = ensemble_simple_average(predictions)

# Weighted average (lower score = higher weight)
scores_dict = {"Holt": 3.2, "Ridge_TS": 1.8, "LSTM_TS": 2.5}
weighted = ensemble_weighted_average(predictions, scores=scores_dict)

# Stacking
stacked = ensemble_stacking(predictions, y_true=y_test)
```

---

## Horizon Strategies

Control how ML models generate multi-step forecasts.

```python
from lazypredict.TimeSeriesForecasting import LazyForecaster

# Recursive (default): single model, autoregressively predicts one step at a time
fcst = LazyForecaster(horizon_strategy="recursive")

# Direct: one separate model trained per forecast horizon step
# More accurate for longer horizons but slower
fcst = LazyForecaster(horizon_strategy="direct")

# Multi-output: MultiOutputRegressor wrapping a single model
fcst = LazyForecaster(horizon_strategy="multi_output")
```

Using horizon functions directly:

```python
from lazypredict.horizon import direct_forecast, multi_output_forecast
from sklearn.linear_model import Ridge

# Direct forecasting
y_pred = direct_forecast(
    estimator_class=Ridge,
    y_train=y_train,
    horizon=20,
    n_lags=10,
)

# Multi-output forecasting
y_pred = multi_output_forecast(
    estimator_class=Ridge,
    y_train=y_train,
    horizon=20,
    n_lags=10,
)
```

---

## Time Series Diagnostics

Run stationarity and statistical tests on your series.

```python
from lazypredict.ts_diagnostics import run_diagnostics

report = run_diagnostics(y_train, seasonal_period=12)
print(report)
# Returns a dict with: adf_statistic, adf_pvalue, kpss_statistic, kpss_pvalue,
# ljung_box_statistic, ljung_box_pvalue, is_stationary, has_seasonality
```

---

## Distributed / PySpark Support

### Auto-convert PySpark/Dask DataFrames

Lazy Predict automatically converts PySpark and Dask DataFrames to pandas
when they are passed to `fit()`:

```python
# PySpark DataFrame — automatically collected to the driver
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
sdf = spark.createDataFrame(df)

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(sdf_train, sdf_test, y_train, y_test)
```

You can also convert explicitly:

```python
from lazypredict.distributed import auto_convert_dataframe

pandas_df = auto_convert_dataframe(spark_df, name="X_train")
```

### Register Dask Joblib Backend

```python
from lazypredict.distributed import register_dask_joblib

register_dask_joblib()  # must have a Dask cluster running
```

---

## Spark MLlib Integration

Benchmark PySpark MLlib models on large distributed datasets.

### Installation

```console
pip install lazypredict[spark]
```

### Usage

```python
from pyspark.sql import SparkSession
from lazypredict.spark import LazySparkClassifier, LazySparkRegressor

spark = SparkSession.builder.appName("LazyPredict").getOrCreate()

# Load data as a Spark DataFrame
train_df = spark.read.csv("train.csv", header=True, inferSchema=True)
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)

feature_cols = [c for c in train_df.columns if c != "label"]

clf = LazySparkClassifier(verbose=1, ignore_warnings=True)
scores = clf.fit(train_df, test_df, feature_cols=feature_cols, label_col="label")
print(scores)
```

Supported Spark MLlib models:

**Classifiers:** LogisticRegression, RandomForestClassifier, GBTClassifier,
DecisionTreeClassifier, LinearSVC, NaiveBayes

**Regressors:** LinearRegression, RandomForestRegressor, GBTRegressor,
DecisionTreeRegressor, GeneralizedLinearRegression

---

## Time Series Tuning

Tune hyperparameters of time series forecasting models directly:

```python
from lazypredict.ts_tuning import tune_forecaster_optuna
from lazypredict.TimeSeriesForecasting import MLForecaster
from sklearn.linear_model import Ridge

# Tune a specific forecaster wrapper
best_params, best_score = tune_forecaster_optuna(
    forecaster_class=MLForecaster,
    y_train=y_train,
    y_val=y_test,
    estimator_class=Ridge,
    n_trials=50,
    timeout=60,
    metric="RMSE",
)
print(f"Best params: {best_params}, Best RMSE: {best_score:.4f}")
```
