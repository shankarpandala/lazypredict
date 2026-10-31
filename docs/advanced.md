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
    tune_timeout=120,       # max 2 minutes per model (no new trial starts after it)
    tune_backend="optuna",  # "optuna", "sklearn", or "flaml"
)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# One row per tuned model: "Best Score" (mean cross-validated score on the
# training data, higher is better), "Best Params" (a dict) and "Status"
print(clf.tuned_scores_)

# Models whose tuning succeeded: best hyperparameters and the pipeline refitted with them
for name, pipeline in clf.tuned_models_.items():
    print(name, clf.tuned_params_[name])

# predict() and explain() use the tuned models with tuned=True, and convert
# NumPy input to the training columns as for the benchmarked models
clf.predict(X_test, tuned=True)
clf.explain(X_test, y_test, tuned=True)
```

Tuning is reproducible for a given `random_state`, uses `cv` folds (5 when `cv`
is not set), and a model that cannot be tuned is reported in the `Status`
column instead of aborting the others. Models without a search space (for
example `LassoCV`, which tunes its own regularisation) are listed as
`"no search space"`.

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

# "Best RMSE" (mean temporal-CV error, lower is better), "Best Params", "Status"
print(fcst.tuned_scores_)
fcst.tuned_models_   # {name: forecaster refitted with the best parameters} for the tuned models
```

### Using the Tuning API Directly

```python
from sklearn.ensemble import RandomForestClassifier

from lazypredict.preprocessing import build_preprocessor, prepare_dataframes
from lazypredict.tuning import tune_supervised_optuna

X_train_df, X_test_df = prepare_dataframes(X_train, X_test)
preprocessor = build_preprocessor(X_train_df, categorical_encoder="onehot")

best_params, best_score = tune_supervised_optuna(
    "RandomForestClassifier",
    RandomForestClassifier,
    X_train_df,
    y_train,
    preprocessor,
    scoring="balanced_accuracy",
    cv=5,
    n_trials=100,
)
```

`tune_supervised_sklearn` (RandomizedSearchCV) and `tune_supervised_flaml`
take the same leading arguments.

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

reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Features as rows, models as columns
importance_df = reg.explain(X_test, y_test, method="permutation", n_repeats=10)
print(importance_df)
```

### SHAP Values

```python
# {model name: SHAP values for the transformed features}
shap_values = reg.explain(X_test, y_test, method="shap", max_samples=100)
shap_values["RandomForestRegressor"].shape   # (n_samples, n_features)
```

Binary classifiers return the values of the positive class; multiclass
classifiers return an array of shape `(n_samples, n_features, n_classes)`.
The lower-level `lazypredict.explainability.explain_shap(models, X_test,
step_name="regressor")` takes the dict of fitted pipelines (`reg.models`).

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

# Inverse-error weighted (lower validation RMSE = higher weight)
weighted_pred = fcst.ensemble(method="weighted_average")

# Stacking (Ridge meta-model)
stacked_pred = fcst.ensemble(method="stacking")
```

Weights and the stacking meta-model are learned on a validation window held
out from the end of the **training** series (each model is refitted on the
rest of `y_train`), never on `y_test`, so the ensemble can be evaluated on
`y_test` like any single model.

!!! note "Changed in 0.4.0"
    `ensemble()` no longer needs `y_true`; passing it emits a `FutureWarning`.
    Earlier versions fitted the stacking model on `y_test`, which leaked the
    test values into the forecast being evaluated.

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

# Stacking: fit the meta-model on the same models' forecasts of a
# validation period (val_predictions vs. y_val), then apply it to the
# forecasts being combined
stacked = ensemble_stacking(val_predictions, y_true=y_val, pred_test=predictions)
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

# With exogenous variables, also pass X_exog (training period) and
# X_future (forecast period).
```

Both raise a `ValueError` when `y_train` is too short to train for the
requested horizon.

---

## Time Series Diagnostics

Check whether a model's residuals look like white noise.

```python
from lazypredict.ts_diagnostics import residual_diagnostics

report = residual_diagnostics(y_test, y_pred, seasonal_period=12)
# dict with: residuals, mean, std, ljung_box_stat, ljung_box_pvalue,
# jarque_bera_stat, jarque_bera_pvalue, acf_values, is_white_noise, is_normal

# Or for every fitted forecaster at once:
diagnostics_df = fcst.diagnose()
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
from lazypredict.distributed import register_dask_backend

register_dask_backend()  # returns False if dask.distributed is unavailable
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

clf = LazySparkClassifier(verbose=1, ignore_warnings=True, label_col="label")
scores = clf.fit(train_df, test_df, feature_cols=feature_cols)
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
from sklearn.linear_model import Ridge

from lazypredict.TimeSeriesForecasting import MLForecaster
from lazypredict.ts_tuning import tune_forecaster_optuna

# Tune a specific forecaster wrapper with expanding-window CV on y_train
best_params, best_score = tune_forecaster_optuna(
    "Ridge_TS",
    MLForecaster(Ridge, "Ridge_TS"),
    y_train,
    X_train=None,
    seasonal_period=12,
    tune_metric="RMSE",
    cv=3,
    n_trials=50,
    timeout=60,
)
print(f"Best params: {best_params}, Best RMSE: {best_score:.4f}")
```
