---
title: 'LazyPredict: Rapid Multi-Model Benchmarking for Machine Learning and Time Series Forecasting in Python'
tags:
  - Python
  - machine learning
  - automated machine learning
  - model selection
  - time series forecasting
  - scikit-learn
authors:
  - name: Shankar Rao Pandala
    orcid: 0009-0006-2234-6944
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 April 2026
bibliography: paper.bib
---

# Summary

LazyPredict is a Python library that enables rapid benchmarking of machine
learning models with minimal code. Given a dataset, it automatically fits and
evaluates over 40 classification and regression models from scikit-learn and
popular gradient boosting libraries, returning a ranked comparison table of
performance metrics. The library extends this approach to time series
forecasting with 26+ models spanning statistical, machine learning, deep
learning, and foundation model categories. LazyPredict provides researchers and
practitioners with a systematic, reproducible way to identify promising model
families before investing effort in hyperparameter tuning, enabling faster
iteration in the model selection phase of machine learning workflows.

# Statement of Need

Model selection is a critical early step in any machine learning project.
Practitioners typically face a large landscape of candidate algorithms and must
decide which model families warrant further investigation. This process is often
ad hoc: researchers manually instantiate a handful of models, fit them one by
one, and compare metrics in a spreadsheet or notebook. This approach is tedious,
error-prone, and frequently biased toward models the practitioner already knows.

Existing AutoML frameworks such as Auto-sklearn [@feurer2015], TPOT [@olson2016],
PyCaret [@pycaret2020], H2O AutoML [@h2o2024], and FLAML [@wu2021] address a
broader optimization problem: they search over model types *and* hyperparameters
simultaneously, often requiring substantial compute time. While powerful, these
tools may be more than what is needed when the goal is simply to survey which
model families perform well on a given dataset before committing to a full
tuning pipeline.

LazyPredict fills this gap by focusing exclusively on rapid, zero-configuration
model benchmarking. With two lines of code, a user obtains a comprehensive
comparison of all applicable models using their default hyperparameters. This
lightweight approach serves several use cases in scientific computing:

- **Exploratory analysis**: Quickly identifying which model families suit a new
  dataset before investing in hyperparameter optimization.
- **Baseline establishment**: Generating a reproducible set of baseline results
  against which tuned or custom models can be compared.
- **Pedagogical use**: Teaching students about the diversity of machine learning
  algorithms and their relative strengths on different data characteristics.
- **Time series model surveys**: Benchmarking 26+ forecasting models, including
  statistical methods (ETS, ARIMA, Theta), ML regressors with lag features,
  deep learning (LSTM, GRU), and foundation models (TimesFM), in a single call.

LazyPredict has been adopted by the research community, accumulating over 45
citations in Google Scholar across diverse domains. Examples include heart
failure prediction in biomedical informatics [@mahgoub2023], spam detection
using word embeddings [@fellah2024], sepsis disease prediction
[@alhashimi2023], and thermal conductivity modeling in chemical engineering
[@varzandeh2025]. The library is available on both PyPI and conda-forge and
supports Python 3.9 through 3.14.

# Software Design and Features

LazyPredict is built around a modular architecture with three main estimator
classes that share a common base class (`LazyEstimator`):

- **`LazyClassifier`**: Benchmarks classification models, reporting accuracy,
  balanced accuracy, ROC AUC, and F1 score for each.
- **`LazyRegressor`**: Benchmarks regression models, reporting adjusted
  R-squared, R-squared, and RMSE.
- **`LazyForecaster`**: Benchmarks time series forecasting models, reporting
  MAE, RMSE, MAPE, SMAPE, MASE, and R-squared.

The library handles preprocessing automatically, including detection and
encoding of categorical features via configurable strategies (one-hot, ordinal,
target, and binary encoding). A scikit-learn `Pipeline` wraps each model with
the appropriate preprocessor, ensuring that all transformations are applied
consistently.

Key architectural decisions include:

- **Graceful degradation**: Models that fail to fit (due to convergence issues,
  incompatible data, or timeouts) are caught and skipped without interrupting
  the full benchmark. Per-model timeout enforcement prevents any single model
  from blocking the run.
- **Optional dependency management**: Heavy dependencies (XGBoost, LightGBM,
  CatBoost, PyTorch, statsmodels, pmdarima, SHAP, MLflow) are optional extras,
  keeping the core installation lightweight.
- **GPU acceleration**: Transparent GPU support for XGBoost, LightGBM, CatBoost,
  cuML (RAPIDS), LSTM/GRU (PyTorch CUDA), and TimesFM, with automatic CPU
  fallback.
- **Extensibility**: Users can pass custom model lists, custom metrics, and
  configure cross-validation folds, enabling adaptation to domain-specific
  evaluation protocols.

Additional modules provide hyperparameter tuning via Optuna and scikit-learn
search (`tuning`), model explainability through permutation importance and SHAP
(`explainability`), ensemble construction from top-performing models
(`ensemble`), and MLflow experiment tracking (`integrations.mlflow`).

# Comparison with Related Software

| Feature | LazyPredict | Auto-sklearn | TPOT | PyCaret | FLAML |
|:--------|:-----------:|:------------:|:----:|:-------:|:-----:|
| Zero-config benchmarking | Yes | No | No | Partial | No |
| Lines of code to compare 40+ models | 2 | 10+ | 10+ | 5+ | 10+ |
| Time series forecasting | Yes (26+ models) | No | No | Yes | Partial |
| Hyperparameter tuning | Optional | Core | Core | Core | Core |
| GPU acceleration | Yes | No | No | Partial | No |
| Default install size | Minimal | Heavy | Heavy | Heavy | Moderate |

LazyPredict is intentionally complementary to full AutoML pipelines: it is
designed for the *survey* phase of model selection, after which users may employ
more compute-intensive tools for the *optimization* phase.

# Acknowledgements

We thank the contributors to the LazyPredict project, including Breno Batista
da Silva, Murat Toprak, Chanin Nantasenamat, and all community members who have
submitted issues and pull requests. We also acknowledge the developers of
scikit-learn, pandas, NumPy, and the broader Python scientific computing
ecosystem on which LazyPredict depends.

# AI Disclosure

Generative AI tools (GitHub Copilot, Claude) were used during development to
assist with code generation, documentation drafting, and test writing. All
AI-generated content was reviewed, tested, and validated by the author.

# References
