---
title: History
---

# 0.3.0a1 (2026-03-10)

-   **New: Time Series Forecasting** — `LazyForecaster` benchmarks 26 forecasting models in one call
-   Statistical models: Naive, SeasonalNaive, SimpleExpSmoothing, Holt, HoltWinters (additive & multiplicative), Theta, SARIMAX, AutoARIMA
-   ML models: LinearRegression, Ridge, Lasso, ElasticNet, KNN, SVR, DecisionTree, RandomForest, GradientBoosting, AdaBoost, Bagging, ExtraTrees, XGBoost, LightGBM
-   Deep learning models: LSTM, GRU (via PyTorch)
-   Foundation model: Google TimesFM 2.5 (200M-parameter zero-shot pretrained transformer)
-   Automatic seasonal period detection via autocorrelation (ACF)
-   Exogenous variable support for SARIMAX, AutoARIMA, and ML models
-   Cross-validation with expanding window (TimeSeriesSplit)
-   New forecasting metrics: MAPE, SMAPE, MASE
-   New install extras: `pip install lazypredict[timeseries]`, `[deeplearning]`, `[foundation]`
-   Added `categorical_encoder` parameter to LazyClassifier and LazyRegressor
-   Refactored Supervised.py with type hints, logging, and input validation

# 0.2.15 (2025-04-06)

-   Added MLflow integration for experiment tracking
-   Added support for Python 3.13
-   Updated all dependencies to latest versions
-   Improved model logging and tracking capabilities
-   Added automatic model signature logging with MLflow

# 0.2.11 (2022-02-06)

-   Updated the default version to 3.9

# 0.2.10 (2022-02-06)

-   Fixed issue with older version of Scikit-learn
-   Reduced dependencies sctrictly to few

# 0.2.8 (2021-02-06)

-   Removed StackingRegressor and CheckingClassifier.
-   Added provided_models method.
-   Added adjusted r-squared metric.
-   Added cardinality check to split categorical columns into low and
    high cardinality features.
-   Added different transformation pipeline for low and high cardinality
    features.
-   Included all number dtypes as inputs.
-   Fixed dependencies.
-   Improved documentation.

# 0.2.7 (2020-07-09)

-   Removed catboost regressor and classifier

# 0.2.6 (2020-01-22)

-   Added xgboost, lightgbm, catboost regressors and classifiers

# 0.2.5 (2020-01-20)

-   Removed troublesome regressors from list of CLASSIFIERS

# 0.2.4 (2020-01-19)

-   Removed troublesome regressors from list of REGRESSORS
-   Added feature to input custom metric for evaluation
-   Added feature to return predictions as dataframe
-   Added model training time for each model

# 0.2.3 (2019-11-22)

-   Removed TheilSenRegressor from list of REGRESSORS
-   Removed GaussianProcessClassifier from list of CLASSIFIERS

# 0.2.2 (2019-11-18)

-   Fixed automatic deployment issue.

# 0.2.1 (2019-11-18)

-   Release of Regression feature.

# 0.2.0 (2019-11-17)

-   Release of Classification feature.

# 0.1.0 (2019-11-16)

-   First release on PyPI.
