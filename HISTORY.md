---
title: History
---

# 0.4.1 (unreleased)

-   `predict()` and `explain()` take `tuned=True` to use `tuned_models_`, with the
    same NumPy-to-DataFrame conversion as the benchmarked models (#631)
-   `datetime64` and `timedelta64` feature columns are used (as days since 1970
    and seconds) instead of being dropped or breaking every model (#632)
-   `tune_timeout` also limits `tune_backend="sklearn"`: no new candidate starts
    after it (#632)

# 0.4.0 (2026-10-30)

The v0.3.1 GitHub release was created on a commit still versioned 0.3.0, so it
never reached PyPI. Its changes (documentation on Material for MkDocs and GitHub
Pages, the JOSS paper files, the model-name filtering fix and Python 3.14
support) are released as part of 0.4.0.

## Breaking changes

-   Python 3.10 or newer is required (3.9 is end-of-life). The minimum versions
    are now scikit-learn 1.2, pandas 1.5, NumPy 1.23 and click 7.0
-   `import lazypredict` no longer calls `sklearnex.patch_sklearn()`. To use
    scikit-learn-intelex, call `patch_sklearn()` yourself before importing
    lazypredict. `INTEL_EXTENSION_AVAILABLE` only reports whether it is
    installed
-   Unknown names in `classifiers=`, `regressors=` and `forecasters=` raise a
    `ValueError` with a close-match suggestion instead of being ignored, and a
    single string selects exactly that model
-   `LazyClassifier` raises a `ValueError` for continuous targets
-   `LazyForecaster.ensemble()` no longer needs `y_true` (passing it emits a
    `FutureWarning`); `ensemble_stacking()` requires `pred_test`
-   `tuned_scores_`: "Best Params" holds dicts and a "Status" column reports
    each model's outcome; the forecaster table's always-NaN "Best Score" column
    is gone (its score column is "Best <tune_metric>")
-   The MLflow integration no longer calls `mlflow.autolog()` or registers a
    model version for every model; each model is logged in a nested run
-   Metrics that cannot be computed are `NaN` instead of `None`, so score
    columns stay numeric

## Release and packaging

-   The publish workflow checks that the release tag matches
    `lazypredict.__version__`, runs the tests, smoke-tests the built wheel, can
    be re-run for an existing tag and attaches the files to the GitHub release.
    The release-please and auto-release workflows, which could not trigger it,
    are removed (#577)
-   The conda recipe builds again with conda 26.3.1 and later (#578)
-   PEP 639 license metadata and a single-sourced version; new `encoders`,
    `test` and `dev` extras replace `requirements*.txt`; the `foundation` extra
    installs `timesfm[torch]` (#581)
-   CI tests the installed package on Python 3.10 and 3.14 for every pull
    request, the OS × Python 3.10–3.14 matrix on dev and weekly, and the
    declared dependency floors; mypy is blocking (#580)
-   The docs preview no longer fails for pull requests from forks, and the API
    reference renders NumPy-style parameter sections (#579)
-   An optional library that is installed but fails to load, such as XGBoost or
    LightGBM on macOS without `libomp`, is skipped with a warning instead of
    making `import lazypredict` fail (#611)
-   Merging a pull request no longer cancels the docs deployment, and PR
    previews are removed when the PR is closed (#612)
-   Running the publish workflow manually for a pushed tag creates the GitHub
    release once the upload to PyPI has succeeded (#613)

## Fixes: LazyClassifier and LazyRegressor

-   pandas `category` columns are encoded instead of silently dropped, and
    `string` columns no longer rely on deprecated pandas behaviour (#582)
-   Cross-validation metrics for classifiers without `predict_proba` (#584), and
    multiclass ROC AUC when the test split lacks a class (#585)
-   `fit()` starts from a clean state, and every pipeline gets its own
    preprocessor (#586)
-   Prediction DataFrames keep the index of `X_test` (#588, contributed by
    AryanRG in #575)
-   Mixed NumPy/DataFrame inputs to `fit()`, `predict()` and `explain()` (#589)
-   `timeout` interrupts slow models (on POSIX, in the main thread) and lists
    them in `errors` (#590)
-   The CLI exits with status 2 when it fails (#592)
-   `tune=True` no longer crashes `fit()`; the scikit-learn backend works for
    every model with a search space and the FLAML backend for the models FLAML
    has a learner for; tuning is seeded, uses `cv`, and exposes `tuned_params_`
    and `tuned_models_` (#604, #605)
-   Adjusted R² is `NaN` instead of dividing by zero or inverting the ranking on
    small test sets; regressors are ranked by adjusted R², then R² (#606)
-   `explain(method="shap")` returns SHAP values (#607)
-   MLflow logging works inside an active run (#608)

## Fixes: LazyForecaster

-   Tuned hyperparameters are applied to the refitted models, and tuning is
    reproducible (#594)
-   The direct and multi-output horizon strategies forecast the right steps;
    `horizon_strategy` and `n_jobs` are honoured (#595)
-   AutoARIMA uses exogenous variables and the seasonal period (#596)
-   Rolling-std features are computed the same way for training and
    forecasting (#597)
-   Misaligned exogenous data and NaN targets raise errors (#598)
-   `predict()` accepts exogenous variables (`X_history`, `X_test`) and no
    longer overwrites the fitted models (#599)
-   Seasonal period detection is robust to trends and noise (#600)
-   Weighted-average and stacking ensembles learn from a validation window of
    the training data instead of the test data (#601)
-   Forecasters are selected by exact name, `fit()` starts from a clean state,
    and `timeout` applies to forecasters (#602)
-   One-column DataFrame targets, `functools.partial` custom metrics and
    case-insensitive `sort_by` work; options are validated (#603)

## Fixes from the pre-release review

-   Holt-Winters models selected by name are skipped with a reason in `errors`
    when there is no seasonal period, instead of raising "unknown name" (#615)
-   `timeout=float("inf")` means no limit, and the SIGALRM handler is always
    restored (#616); `n_jobs=0` is rejected (#617); ensembles work for a
    one-step test horizon (#618)
-   Seasonal-period detection finds seasonality from three cycles and no longer
    reports periods for curved trends (#623)
-   The direct and multi-output strategies use every future exogenous row
    (#624); the stacking ensemble no longer depends on the series' units (#625)
-   Equal numeric category codes match across splits (1 vs 1.0) (#620); object
    NumPy arrays keep numeric columns numeric, and integer column labels work
    (#621)
-   The FLAML backend reports defaults as a valid result, keeps CatBoost and
    LightGBM parameters, and the `flaml` extra installs `flaml[automl]` (#622)
-   MLflow tracking problems no longer change benchmark results, and the
    estimator's hyperparameters are logged (#626)
-   `explain(method="shap")` explains classifiers through their probabilities
    (#627)
-   Forecaster tuning works for every horizon strategy and metric (#628)
-   Smaller fixes: one-column targets, DataFrames after a NumPy fit, the CLI
    exit status and seed, forecast indexes, `sort_by` validation, the publish
    workflow's re-runs and smoke test (#619, #629); documentation (#630)

## Documentation

-   Fixed examples and API references, regenerated the README result tables
    and added the missing 0.3.0 changelog entries (#609)
-   YAML issue forms and a pull request template; removed stale planning
    documents (#610)

# 0.3.0 (2026-03-15)

-   First stable release of the 0.3 series: time series forecasting, hyperparameter
    tuning, explainability, GPU support and the integrations listed in the
    pre-release notes below
-   Import `torch` and `timesfm` lazily to avoid OpenMP deadlocks with scikit-learn

# 0.3.0a5 (2026-03-14)

-   Added time series visualization (`LazyForecaster.plot_results()`,
    `lazypredict.ts_visualization`) and residual diagnostics
    (`LazyForecaster.diagnose()`, `lazypredict.ts_diagnostics`)
-   Forecaster tuning refits the tuned models (`tuned_models_`)
-   Raised test coverage to 67%; fixed flake8 and CI failures

# 0.3.0a4 (2026-03-10)

-   Implemented explainability: `explain()` method with permutation importance and SHAP
-   Added Optuna/sklearn/FLAML hyperparameter tuning backends (`tune=True`)
-   Added search spaces for 30+ supervised models and 20+ forecasting models
-   Added forecasting optimization: direct/multi-output horizon strategies, ensemble methods
-   Added seasonal period tuning and forecast loss function selection
-   Added EBM (ExplainableBoostingMachine) from InterpretML
-   Added PySpark/Dask DataFrame auto-conversion
-   Added Dask joblib backend for distributed cross-validation
-   Added Spark MLlib integration: `LazySparkClassifier` / `LazySparkRegressor`
-   Added FLAML as optional tuning backend
-   New optional extras: `tune`, `explain`, `interpret`, `flaml`, `spark`

# 0.3.0a3 (2026-03-10)

-   Added `use_gpu` parameter to classification and regression advanced options docs
-   Added `foundation_model_path` parameter to LazyForecaster for offline/air-gapped environments
-   Added CatBoost to `boost` install extras (`pip install lazypredict[boost]`)
-   Updated all documentation examples to include GPU acceleration and local model weights
-   Fixed Sphinx documentation build pipeline (RST formatting, myst_parser migration)
-   Updated ReadTheDocs and GitHub Actions docs workflow configuration

# 0.3.0a1 (2026-03-10)

-   **New: Time Series Forecasting** — `LazyForecaster` benchmarks 26+ forecasting models in one call
-   Statistical models: Naive, SeasonalNaive, SimpleExpSmoothing, Holt, HoltWinters (additive & multiplicative), Theta, SARIMAX, AutoARIMA
-   ML models: LinearRegression, Ridge, Lasso, ElasticNet, KNN, SVR, DecisionTree, RandomForest, GradientBoosting, AdaBoost, Bagging, ExtraTrees, XGBoost, LightGBM, CatBoost
-   Deep learning models: LSTM, GRU (via PyTorch)
-   Foundation model: Google TimesFM 2.5 (200M-parameter zero-shot pretrained transformer)
-   **GPU acceleration** via `use_gpu=True`: XGBoost (`device="cuda"`), LightGBM (`device="gpu"`), CatBoost (`task_type="GPU"`), cuML (RAPIDS) auto-discovery, LSTM/GRU/TimesFM CUDA device placement
-   Automatic seasonal period detection via autocorrelation (ACF)
-   Exogenous variable support for SARIMAX, AutoARIMA, and ML models
-   Cross-validation with expanding window (TimeSeriesSplit)
-   New forecasting metrics: MAPE, SMAPE, MASE
-   New install extras: `pip install lazypredict[timeseries]`, `[deeplearning]`, `[foundation]`
-   Added `categorical_encoder` parameter to LazyClassifier and LazyRegressor
-   Added CatBoost models (CatBoostClassifier, CatBoostRegressor) to supervised and time series
-   Added cuML (RAPIDS) GPU-native scikit-learn model auto-discovery when `use_gpu=True`
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
