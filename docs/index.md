# Welcome to Lazy Predict

**Lazy Predict** helps you build and benchmark a large number of machine learning models with minimal code, so you can quickly discover which algorithms work best on your dataset — without any parameter tuning.

[![PyPI version](https://img.shields.io/pypi/v/lazypredict.svg)](https://pypi.python.org/pypi/lazypredict)
[![Downloads](https://pepy.tech/badge/lazypredict)](https://pepy.tech/project/lazypredict)
[![CodeFactor](https://www.codefactor.io/repository/github/shankarpandala/lazypredict/badge)](https://www.codefactor.io/repository/github/shankarpandala/lazypredict)

---

## Quick Start

=== "Classification"

    ```python
    from lazypredict.Supervised import LazyClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    ```

=== "Regression"

    ```python
    from lazypredict.Supervised import LazyRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(models)
    ```

=== "Time Series"

    ```python
    import numpy as np
    from lazypredict.TimeSeriesForecasting import LazyForecaster

    t = np.arange(200)
    y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 200)

    fcst = LazyForecaster(verbose=0, ignore_warnings=True)
    scores, predictions = fcst.fit(y[:180], y[180:])
    print(scores)
    ```

---

## What Can It Do?

| Task | Class | Models |
|------|-------|--------|
| Classification | `LazyClassifier` | 40+ scikit-learn classifiers + XGBoost, LightGBM, CatBoost, cuML |
| Regression | `LazyRegressor` | 40+ scikit-learn regressors + XGBoost, LightGBM, CatBoost, cuML |
| Time Series | `LazyForecaster` | 20+ statistical, ML, deep learning, and foundation models |
| Distributed | `LazySparkClassifier` / `LazySparkRegressor` | PySpark MLlib models |

---

## Feature Highlights

- **Zero-boilerplate benchmarking** — one call trains and evaluates all models
- **40+ classifiers and regressors** from scikit-learn plus optional boosting libraries
- **20+ time series forecasting models** — Naive, ETS, ARIMA, Theta, ML, LSTM/GRU, TimesFM
- **GPU acceleration** — XGBoost, LightGBM, CatBoost, cuML (RAPIDS), LSTM/GRU, TimesFM
- **Automatic seasonal period detection** via autocorrelation analysis
- **Exogenous variables** support in time series forecasting (SARIMAX, AutoARIMA, ML models)
- **Multiple categorical encoders** — OneHot, Ordinal, Target, Binary
- **MLflow integration** — automatic experiment tracking via environment variable
- **Cross-validation** — both train/test split and k-fold CV supported
- **Per-model timeout** — skip slow models automatically
- **Hyperparameter tuning** — Optuna, scikit-learn, or FLAML backends
- **Explainability** — permutation importance and optional SHAP values
- **Ensemble methods** — simple average, weighted average, stacking
- **Visualization** — forecast plots, model comparison, residuals, error heatmaps
- **Intel Extension acceleration** — automatic CPU speedup when installed
- **Python 3.9 – 3.13** support
- **MIT licensed**, free software

---

## Getting Started

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Install from PyPI, conda-forge, or source. Learn about optional extras.

-   :material-book-open: **[Usage Guide](usage.md)**

    Core usage patterns for classification, regression, time series, and the CLI.

-   :material-chart-line: **[Time Series Forecasting](forecasting.md)**

    Deep dive into `LazyForecaster` — all 20+ models, cross-validation, ensembles, and plots.

-   :material-beaker: **[Examples](examples.md)**

    Detailed worked examples for every feature with copy-paste code.

-   :material-cog: **[Advanced Features](advanced.md)**

    Explainability, hyperparameter tuning, visualization, distributed computing, and Spark.

-   :material-code-tags: **[API Reference](api/lazypredict.md)**

    Complete auto-generated API docs for every public class and function.

</div>

---

## Links

- [Source Code](https://github.com/shankarpandala/lazypredict)
- [PyPI](https://pypi.org/project/lazypredict/)
- [Bug Reports / Feature Requests](https://github.com/shankarpandala/lazypredict/issues)
