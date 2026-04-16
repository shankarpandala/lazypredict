---
title: History
---

# 0.3.0 (2026-03-15)

-   **New: Time Series Forecasting** — `LazyForecaster` benchmarks 26+ forecasting models in one call
    -   Statistical models: Naive, SeasonalNaive, SimpleExpSmoothing, Holt, HoltWinters (additive & multiplicative), Theta, SARIMAX, AutoARIMA
    -   ML models: LinearRegression, Ridge, Lasso, ElasticNet, KNN, SVR, DecisionTree, RandomForest, GradientBoosting, AdaBoost, Bagging, ExtraTrees, XGBoost, LightGBM, CatBoost
    -   Deep learning models: LSTM, GRU (via PyTorch)
    -   Foundation model: Google TimesFM (200M-parameter zero-shot pretrained transformer)
-   **GPU acceleration** via `use_gpu=True`: XGBoost, LightGBM, CatBoost, cuML (RAPIDS), LSTM/GRU, TimesFM
-   Automatic seasonal period detection via autocorrelation (ACF)
-   Exogenous variable support for SARIMAX, AutoARIMA, and ML models
-   Cross-validation with expanding window (TimeSeriesSplit)
-   New forecasting metrics: MAPE, SMAPE, MASE
-   Added `categorical_encoder` parameter to LazyClassifier and LazyRegressor (onehot, ordinal, target, binary)
-   Added CatBoost models to supervised and time series
-   Added cuML (RAPIDS) GPU-native scikit-learn model auto-discovery
-   Implemented explainability: `explain()` with permutation importance and SHAP
-   Added Optuna/sklearn/FLAML hyperparameter tuning backends
-   Added search spaces for 30+ supervised models and 20+ forecasting models
-   Added forecasting optimization: direct/multi-output horizon strategies, ensemble methods
-   Added EBM (ExplainableBoostingMachine) from InterpretML
-   Added PySpark/Dask DataFrame auto-conversion
-   Added Spark MLlib integration: `LazySparkClassifier` / `LazySparkRegressor`
-   New install extras: `timeseries`, `deeplearning`, `foundation`, `boost`, `tune`, `explain`, `interpret`, `flaml`, `spark`, `all`
-   Refactored Supervised.py into modular architecture with `LazyEstimator` base class, type hints, logging, and input validation
-   Migrated build configuration from setup.py to pyproject.toml
-   Updated development status from Alpha to Beta

# 0.2.16 (2025-04-05)

-   Patch release following 0.2.15 version fixes

# 0.2.15 (2025-04-01)

-   Added MLflow integration for experiment tracking
-   Added support for Python 3.13
-   Updated all dependencies to latest versions
-   Added automatic model signature logging with MLflow

# 0.2.13 (2024-11-01)

-   Migrated CI from Travis CI to GitHub Actions
-   Added conda-forge recipe and conda publishing
-   Replaced deprecated Boston dataset with Diabetes dataset in examples
-   Added unit tests for CLI and supervised learning components
-   Updated README badges and documentation

# 0.2.12 (2022-09-28)

-   Updated default Python version to 3.10
-   Reduced dependencies strictly to core requirements
-   Fixed issue with older versions of scikit-learn
-   Updated CI configuration
-   Added click to requirements

# 0.2.9 (2021-02-17)

-   Minor maintenance release

# 0.2.8 (2021-02-06)

-   Removed StackingRegressor and CheckingClassifier
-   Added `provided_models` method
-   Added adjusted R-squared metric
-   Added cardinality check to split categorical columns into low and high cardinality features
-   Added different transformation pipeline for low and high cardinality features
-   Included all number dtypes as inputs
-   Fixed dependencies
-   Improved documentation

# 0.2.7 (2020-07-09)

-   Removed catboost regressor and classifier (dependency issues)

# 0.2.6 (2020-01-22)

-   Added XGBoost, LightGBM, CatBoost regressors and classifiers

# 0.2.5 (2020-01-20)

-   Removed troublesome classifiers from list of CLASSIFIERS

# 0.2.4 (2020-01-19)

-   Removed troublesome regressors from list of REGRESSORS
-   Added feature to input custom metric for evaluation
-   Added feature to return predictions as DataFrame
-   Added model training time for each model

# 0.2.3 (2019-11-22)

-   Removed TheilSenRegressor from list of REGRESSORS
-   Removed GaussianProcessClassifier from list of CLASSIFIERS

# 0.2.2 (2019-11-18)

-   Fixed automatic deployment issue

# 0.2.1 (2019-11-18)

-   Release of Regression feature

# 0.2.0 (2019-11-17)

-   Release of Classification feature

# 0.1.0 (2019-11-16)

-   First release on PyPI
