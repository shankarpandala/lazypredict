# Welcome to Lazy Predict

Lazy Predict helps build a lot of basic models without much code and helps understand which models work better without any parameter tuning.

[![PyPI version](https://img.shields.io/pypi/v/lazypredict.svg)](https://pypi.python.org/pypi/lazypredict)
[![Downloads](https://pepy.tech/badge/lazypredict)](https://pepy.tech/project/lazypredict)
[![CodeFactor](https://www.codefactor.io/repository/github/shankarpandala/lazypredict/badge)](https://www.codefactor.io/repository/github/shankarpandala/lazypredict)

## Quick Links

- [Source Code](https://github.com/shankarpandala/lazypredict)
- [Bug Reports](https://github.com/shankarpandala/lazypredict/issues)
- [Documentation](https://shankarpandala.github.io/lazypredict/)

## Features

- Over 40 built-in machine learning models
- Automatic model selection for classification, regression, and **time series forecasting**
- **20+ forecasting models** — statistical, ML, deep learning, and pretrained foundation models
- **GPU acceleration** — XGBoost, LightGBM, CatBoost, cuML (RAPIDS), LSTM/GRU, TimesFM
- Automatic seasonal period detection via autocorrelation
- Support for univariate series with optional exogenous variables
- Support for both numerical and categorical features
- Easy integration with scikit-learn pipelines
- Model performance comparison and ranking
- Built-in MLflow integration for experiment tracking
- Support for Python 3.9 through 3.13
- Minimal code required
- Automatic model metrics logging
- Custom metric evaluation support
- Easy model access and reuse

## Key Benefits

- Rapid model prototyping and selection
- One-line benchmarking of forecasting models (`LazyForecaster`)
- GPU acceleration with automatic fallback to CPU
- Automated experiment tracking with MLflow
- Comprehensive model performance comparison
- Zero-configuration model evaluation
- Support for local and remote tracking
- Integration with Databricks environment
- Parallel model training capability
- Extensible with custom metrics

## Getting Started

- [Installation](installation.md)
- [Usage](usage.md)
- [Examples](examples.md)
- [API Reference](api/lazypredict.md)
