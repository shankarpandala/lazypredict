# Lazy Predict

[![image](https://img.shields.io/pypi/v/lazypredict.svg)](https://pypi.python.org/pypi/lazypredict)
[![Publish](https://github.com/shankarpandala/lazypredict/actions/workflows/publish.yml/badge.svg)](https://github.com/shankarpandala/lazypredict/actions/workflows/publish.yml)
[![Documentation](https://github.com/shankarpandala/lazypredict/actions/workflows/docs.yml/badge.svg)](https://shankarpandala.github.io/lazypredict/)
[![Downloads](https://pepy.tech/badge/lazypredict)](https://pepy.tech/project/lazypredict)
[![CodeFactor](https://www.codefactor.io/repository/github/shankarpandala/lazypredict/badge)](https://www.codefactor.io/repository/github/shankarpandala/lazypredict)
[![Citations](https://img.shields.io/badge/Citations-43-blue)](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=4325808232671020176,16284230108871951652&as_sdt=5)

Lazy Predict helps build a lot of basic models without much code and helps understand which models work better without any parameter tuning.

- Free software: MIT license
- Documentation: <https://shankarpandala.github.io/lazypredict/>

## Features
- Over 40 built-in machine learning models
- Automatic model selection for classification, regression, and **time series forecasting**
- **20+ forecasting models**: statistical (ETS, ARIMA, Theta), ML (Random Forest, XGBoost, etc.), deep learning (LSTM, GRU), and pretrained foundation models (TimesFM)
- Automatic seasonal period detection via ACF
- Multiple categorical encoding strategies (OneHot, Ordinal, Target, Binary)
- Built-in MLflow integration for experiment tracking
- **GPU acceleration**: XGBoost, LightGBM, CatBoost, cuML (RAPIDS), LSTM/GRU, TimesFM
- Support for Python 3.10 through 3.14
- Custom metric evaluation support
- Configurable timeout and cross-validation
- Intel Extension for Scikit-learn acceleration support (opt-in)

## Installation

### pip (PyPI)

```bash
pip install lazypredict
```

### conda (conda-forge)

```bash
conda install -c conda-forge lazypredict
```

### Optional extras (pip only)

Install with boosting libraries (XGBoost, LightGBM, CatBoost):

```bash
pip install lazypredict[boost]
```

Install the target/binary categorical encoders (`category-encoders`):

```bash
pip install lazypredict[encoders]
```

Install with time series forecasting support:

```bash
pip install lazypredict[timeseries]          # statsmodels + pmdarima
pip install lazypredict[timeseries,deeplearning]  # + LSTM/GRU via PyTorch
pip install lazypredict[timeseries,foundation]    # + Google TimesFM (PyTorch)
```

Install with all optional dependencies:

```bash
pip install lazypredict[all]
```

## Usage

To use Lazy Predict in a project:

```python
import lazypredict
```

## Classification

Example:

```python
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
```

### Advanced Options

```python
# With categorical encoding, timeout, cross-validation, and GPU
clf = LazyClassifier(
    verbose=1,                          # Show progress
    ignore_warnings=True,               # Suppress warnings
    custom_metric=None,                 # Use default metrics
    predictions=True,                   # Return predictions
    classifiers='all',                  # Use all available classifiers
    categorical_encoder='onehot',       # Encoding: 'onehot', 'ordinal', 'target', 'binary'
    timeout=60,                         # Max time per model in seconds
    cv=5,                               # Cross-validation folds (optional)
    use_gpu=True                        # Enable GPU acceleration
)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

**Parameters:**
- `verbose` (int): 0 for silent, 1 for progress display
- `ignore_warnings` (bool): Suppress scikit-learn warnings
- `custom_metric` (callable): Custom evaluation metric
- `predictions` (bool): Return prediction DataFrame
- `classifiers` (str/list): 'all' or list of classifier names
- `categorical_encoder` (str): Encoding strategy for categorical features
  - `'onehot'`: One-hot encoding (default)
  - `'ordinal'`: Ordinal encoding
  - `'target'`: Target encoding (requires `category-encoders`)
  - `'binary'`: Binary encoding (requires `category-encoders`)
- `timeout` (int/float): Maximum training time per model in seconds (None for no limit). Slow models are interrupted on Linux/macOS and listed in `errors` as `TimeoutException`
- `cv` (int): Number of cross-validation folds (None to disable)
- `use_gpu` (bool): Enable GPU acceleration for supported models (default False)

Output on the breast cancer split above (scikit-learn 1.9, with `xgboost` and `lightgbm` installed; times are machine-dependent):

| Model                       | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Precision | Recall | Time Taken |
|:----------------------------|---------:|------------------:|--------:|---------:|----------:|-------:|-----------:|
| LinearSVC                   |   0.9895 |            0.9875 |  0.9925 |   0.9895 |    0.9895 | 0.9895 |       0.02 |
| Perceptron                  |   0.9860 |            0.9848 |  0.9932 |   0.9860 |    0.9860 | 0.9860 |       0.04 |
| LogisticRegression          |   0.9860 |            0.9827 |  0.9947 |   0.9859 |    0.9860 | 0.9860 |       0.04 |
| SVC                         |   0.9825 |            0.9799 |  0.9949 |   0.9824 |    0.9824 | 0.9825 |       0.03 |
| XGBClassifier               |   0.9754 |            0.9744 |  0.9874 |   0.9755 |    0.9755 | 0.9754 |       0.07 |
| LabelSpreading              |   0.9754 |            0.9744 |         |   0.9755 |    0.9755 | 0.9754 |       0.04 |
| LabelPropagation            |   0.9754 |            0.9744 |         |   0.9755 |    0.9755 | 0.9754 |       0.03 |
| BaggingClassifier           |   0.9719 |            0.9738 |  0.9860 |   0.9720 |    0.9726 | 0.9719 |       0.07 |
| PassiveAggressiveClassifier |   0.9789 |            0.9730 |  0.9962 |   0.9789 |    0.9792 | 0.9789 |       0.04 |
| SGDClassifier               |   0.9754 |            0.9723 |  0.9918 |   0.9754 |    0.9754 | 0.9754 |       0.03 |
| RandomForestClassifier      |   0.9719 |            0.9717 |  0.9933 |   0.9720 |    0.9721 | 0.9719 |       0.21 |
| CalibratedClassifierCV      |   0.9789 |            0.9709 |  0.9939 |   0.9788 |    0.9796 | 0.9789 |       0.07 |
| AdaBoostClassifier          |   0.9719 |            0.9696 |  0.9906 |   0.9719 |    0.9719 | 0.9719 |       0.18 |
| LGBMClassifier              |   0.9684 |            0.9668 |  0.9895 |   0.9685 |    0.9685 | 0.9684 |       0.07 |
| ExtraTreesClassifier        |   0.9684 |            0.9647 |  0.9949 |   0.9684 |    0.9684 | 0.9684 |       0.15 |
| RidgeClassifierCV           |   0.9719 |            0.9633 |  0.9924 |   0.9717 |    0.9725 | 0.9719 |       0.03 |
| RidgeClassifier             |   0.9684 |            0.9605 |  0.9919 |   0.9682 |    0.9687 | 0.9684 |       0.03 |
| KNeighborsClassifier        |   0.9614 |            0.9550 |  0.9866 |   0.9613 |    0.9614 | 0.9614 |       0.04 |
| BernoulliNB                 |   0.9509 |            0.9510 |  0.9899 |   0.9511 |    0.9517 | 0.9509 |       0.03 |
| LinearDiscriminantAnalysis  |   0.9614 |            0.9508 |  0.9823 |   0.9611 |    0.9621 | 0.9614 |       0.03 |
| GaussianNB                  |   0.9544 |            0.9495 |  0.9935 |   0.9543 |    0.9543 | 0.9544 |       0.02 |
| NuSVC                       |   0.9544 |            0.9432 |  0.9918 |   0.9540 |    0.9549 | 0.9544 |       0.05 |
| ExtraTreeClassifier         |   0.9439 |            0.9350 |  0.9350 |   0.9436 |    0.9438 | 0.9439 |       0.03 |
| NearestCentroid             |   0.9474 |            0.9335 |  0.9891 |   0.9468 |    0.9483 | 0.9474 |       0.04 |
| DecisionTreeClassifier      |   0.9298 |            0.9282 |  0.9282 |   0.9301 |    0.9308 | 0.9298 |       0.03 |
| DummyClassifier             |   0.6386 |            0.5000 |  0.5000 |   0.4977 |    0.4078 | 0.6386 |       0.03 |

## Regression

Example:

```python
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

diabetes  = datasets.load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
```

### Advanced Options

```python
# With categorical encoding, timeout, and GPU
reg = LazyRegressor(
    verbose=1,                          # Show progress
    ignore_warnings=True,               # Suppress warnings
    custom_metric=None,                 # Use default metrics
    predictions=True,                   # Return predictions
    regressors='all',                   # Use all available regressors
    categorical_encoder='ordinal',      # Encoding: 'onehot', 'ordinal', 'target', 'binary'
    timeout=120,                        # Max time per model in seconds
    use_gpu=True                        # Enable GPU acceleration
)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
```

**Parameters:**
- `verbose` (int): 0 for silent, 1 for progress display
- `ignore_warnings` (bool): Suppress scikit-learn warnings
- `custom_metric` (callable): Custom evaluation metric
- `predictions` (bool): Return prediction DataFrame
- `regressors` (str/list): 'all' or list of regressor names
- `categorical_encoder` (str): Encoding strategy for categorical features
  - `'onehot'`: One-hot encoding (default)
  - `'ordinal'`: Ordinal encoding
  - `'target'`: Target encoding (requires `category-encoders`)
  - `'binary'`: Binary encoding (requires `category-encoders`)
- `timeout` (int/float): Maximum training time per model in seconds (None for no limit). Slow models are interrupted on Linux/macOS and listed in `errors` as `TimeoutException`
- `use_gpu` (bool): Enable GPU acceleration for supported models (default False)

Output on the diabetes split above:

| Model                         | Adjusted R-Squared | R-Squared |     RMSE | Time Taken |
|:------------------------------|-------------------:|----------:|---------:|-----------:|
| ExtraTreesRegressor           |             0.3789 |    0.5201 |  54.2202 |       0.16 |
| OrthogonalMatchingPursuitCV   |             0.3749 |    0.5170 |  54.3934 |       0.02 |
| Lasso                         |             0.3735 |    0.5159 |  54.4570 |       0.01 |
| LassoLars                     |             0.3735 |    0.5159 |  54.4574 |       0.02 |
| LarsCV                        |             0.3715 |    0.5143 |  54.5431 |       0.04 |
| LassoCV                       |             0.3704 |    0.5135 |  54.5903 |       0.12 |
| PassiveAggressiveRegressor    |             0.3670 |    0.5108 |  54.7399 |       0.01 |
| LassoLarsIC                   |             0.3650 |    0.5093 |  54.8251 |       0.02 |
| SGDRegressor                  |             0.3643 |    0.5088 |  54.8544 |       0.01 |
| Ridge                         |             0.3630 |    0.5078 |  54.9107 |       0.01 |
| RidgeCV                       |             0.3630 |    0.5078 |  54.9107 |       0.01 |
| BayesianRidge                 |             0.3623 |    0.5072 |  54.9411 |       0.01 |
| LassoLarsCV                   |             0.3617 |    0.5068 |  54.9646 |       0.03 |
| LinearRegression              |             0.3617 |    0.5068 |  54.9646 |       0.01 |
| TransformedTargetRegressor    |             0.3617 |    0.5068 |  54.9646 |       0.01 |
| Lars                          |             0.3588 |    0.5045 |  55.0903 |       0.04 |
| ElasticNetCV                  |             0.3562 |    0.5025 |  55.2048 |       0.07 |
| HuberRegressor                |             0.3553 |    0.5018 |  55.2437 |       0.04 |
| RandomForestRegressor         |             0.3478 |    0.4960 |  55.5630 |       0.32 |
| AdaBoostRegressor             |             0.3404 |    0.4903 |  55.8757 |       0.08 |
| LGBMRegressor                 |             0.3392 |    0.4894 |  55.9255 |       0.04 |
| PoissonRegressor              |             0.3230 |    0.4768 |  56.6103 |       0.98 |
| HistGradientBoostingRegressor |             0.3141 |    0.4699 |  56.9814 |       0.18 |
| ElasticNet                    |             0.3018 |    0.4604 |  57.4899 |       0.01 |
| KNeighborsRegressor           |             0.2999 |    0.4590 |  57.5681 |       0.02 |
| OrthogonalMatchingPursuit     |             0.2924 |    0.4532 |  57.8729 |       0.01 |
| BaggingRegressor              |             0.2779 |    0.4420 |  58.4654 |       0.04 |
| XGBRegressor                  |             0.2505 |    0.4209 |  59.5610 |       0.09 |
| TweedieRegressor              |             0.2442 |    0.4160 |  59.8118 |       0.17 |
| GradientBoostingRegressor     |             0.2362 |    0.4098 |  60.1296 |       0.15 |
| GammaRegressor                |             0.2239 |    0.4003 |  60.6105 |       0.05 |
| RANSACRegressor               |             0.2035 |    0.3845 |  61.4004 |       0.12 |
| LinearSVR                     |             0.1167 |    0.3175 |  64.6607 |       0.02 |
| ExtraTreeRegressor            |             0.0020 |    0.2288 |  68.7304 |       0.01 |
| NuSVR                         |            -0.0667 |    0.1757 |  71.0575 |       0.02 |
| SVR                           |            -0.0964 |    0.1528 |  72.0402 |       0.02 |
| DummyRegressor                |            -0.2976 |   -0.0027 |  78.3701 |       0.01 |
| QuantileRegressor             |            -0.3466 |   -0.0405 |  79.8364 |       0.03 |
| DecisionTreeRegressor         |            -0.4698 |   -0.1358 |  83.4105 |       0.01 |
| GaussianProcessRegressor      |            -0.7692 |   -0.3671 |  91.5109 |       0.07 |
| MLPRegressor                  |            -1.8677 |   -1.2160 | 116.5081 |       0.29 |
| KernelRidge                   |            -5.0382 |   -3.6659 | 169.0604 |       0.05 |

## Time Series Forecasting

LazyForecaster benchmarks 20+ forecasting models on your time series in a single call:

```python
import numpy as np
from lazypredict.TimeSeriesForecasting import LazyForecaster

# Generate sample data (or use your own)
np.random.seed(42)
t = np.arange(200)
rng = np.random.default_rng(42)
y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 200)

y_train, y_test = y[:180], y[180:]

fcst = LazyForecaster(verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)
print(scores)
```

Output with `lazypredict[timeseries,boost]` installed (seasonal period 12 detected; times are machine-dependent):

| Model               |    MAE |   RMSE |   MAPE |  SMAPE |   MASE | R-Squared | Time Taken |
|:--------------------|-------:|-------:|-------:|-------:|-------:|----------:|-----------:|
| SARIMAX             | 0.8569 | 1.0778 | 4.3024 | 4.3970 | 0.7765 |    0.7433 |       0.38 |
| HoltWinters_Add     | 0.8759 | 1.0908 | 4.4003 | 4.4860 | 0.7937 |    0.7371 |       0.09 |
| HoltWinters_Mul     | 0.9515 | 1.2776 | 4.8192 | 4.9717 | 0.8622 |    0.6393 |       0.08 |
| AutoARIMA           | 1.0586 | 1.3570 | 5.3003 | 5.3961 | 0.9593 |    0.5931 |      11.65 |
| LinearRegression_TS | 1.1409 | 1.3578 | 5.5761 | 5.7176 | 1.0339 |    0.5926 |       0.02 |
| Ridge_TS            | 1.1716 | 1.3883 | 5.7135 | 5.8667 | 1.0617 |    0.5741 |       0.01 |
| ...                 |    ... |    ... |    ... |    ... |    ... |       ... |        ... |

### With Exogenous Variables

```python
# Optional exogenous features
X_train = np.column_stack([np.sin(t[:180]), np.cos(t[:180])])
X_test = np.column_stack([np.sin(t[180:]), np.cos(t[180:])])

scores, predictions = fcst.fit(y_train, y_test, X_train, X_test)
```

### Advanced Options

```python
fcst = LazyForecaster(
    verbose=1,                          # Show progress
    ignore_warnings=True,               # Suppress model errors
    predictions=True,                   # Return forecast values
    seasonal_period=12,                 # Override auto-detection
    cv=3,                               # Time series cross-validation
    timeout=30,                         # Max seconds per model
    sort_by="RMSE",                     # Sort metric (MAE, MAPE, SMAPE, MASE, R-Squared)
    forecasters="all",                  # Or a list, e.g. ["Holt", "Ridge_TS"] (Holt needs [timeseries])
    max_models=10,                      # Limit number of models
    use_gpu=True,                       # GPU acceleration for supported models
    foundation_model_path="/path/to/timesfm-weights",  # Local model weights (offline)
)
scores, predictions = fcst.fit(y_train, y_test)
```

**Parameters:**
- `verbose` (int): 0 for silent, 1 for progress display
- `ignore_warnings` (bool): Suppress per-model exceptions
- `predictions` (bool): Return a second DataFrame of forecasted values
- `seasonal_period` (int/None): Seasonal cycle length; ``None`` auto-detects via ACF
- `cv` (int/None): Number of ``TimeSeriesSplit`` folds for cross-validation
- `timeout` (int/float/None): Maximum seconds per model for training and forecasting; slow models are interrupted on Linux/macOS and listed in `errors`
- `sort_by` (str): Metric to sort by (``"RMSE"``, ``"MAE"``, ``"MAPE"``, ``"SMAPE"``, ``"MASE"``, ``"R-Squared"``)
- `forecasters` (str/list): ``"all"`` or a list of model names
- `n_lags` (int): Number of lag features for ML/DL models (default 10)
- `n_rolling` (tuple): Rolling-window sizes for feature engineering (default (3, 7))
- `max_models` (int/None): Limit total models to train
- `custom_metric` (callable): Additional metric ``f(y_true, y_pred) -> float``
- `use_gpu` (bool): Enable GPU acceleration for supported models (default False)
- `foundation_model_path` (str): Local path to pre-downloaded foundation model weights (e.g. TimesFM)

**Available model categories:**
- **Baselines:** Naive, SeasonalNaive
- **Statistical (statsmodels):** SimpleExpSmoothing, Holt, HoltWinters_Add, HoltWinters_Mul, Theta, SARIMAX
- **Statistical (pmdarima):** AutoARIMA
- **ML (sklearn):** LinearRegression_TS, Ridge_TS, Lasso_TS, ElasticNet_TS, KNeighborsRegressor_TS, DecisionTreeRegressor_TS, RandomForestRegressor_TS, GradientBoostingRegressor_TS, AdaBoostRegressor_TS, ExtraTreesRegressor_TS, BaggingRegressor_TS, SVR_TS, XGBRegressor_TS, LGBMRegressor_TS, CatBoostRegressor_TS
- **Deep Learning (torch):** LSTM_TS, GRU_TS
- **Foundation (timesfm):** TimesFM

## GPU Acceleration

Enable GPU acceleration for supported models with `use_gpu=True`:

```python
from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Classification with GPU
clf = LazyClassifier(use_gpu=True, verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Regression with GPU
reg = LazyRegressor(use_gpu=True, verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Time Series with GPU
from lazypredict.TimeSeriesForecasting import LazyForecaster
fcst = LazyForecaster(use_gpu=True, verbose=0, ignore_warnings=True)
scores, predictions = fcst.fit(y_train, y_test)
```

**Supported GPU backends:**
- **XGBoost** — `device="cuda"`
- **LightGBM** — `device="gpu"`
- **CatBoost** — `task_type="GPU"`
- **cuML (RAPIDS)** — GPU-native scikit-learn replacements (auto-discovered when installed)
- **LSTM / GRU** — PyTorch CUDA
- **TimesFM** — PyTorch CUDA

Falls back to CPU automatically if no CUDA GPU is available.

## Categorical Encoding

Lazy Predict supports multiple categorical encoding strategies:

```python
from lazypredict.Supervised import LazyClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Example with categorical features
df = pd.read_csv('data_with_categories.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Try different encoders
for encoder in ['onehot', 'ordinal', 'target', 'binary']:
    clf = LazyClassifier(
        categorical_encoder=encoder,
        verbose=0,
        ignore_warnings=True
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\n{encoder.upper()} Encoding Results:")
    print(models.head())
```

**Note:** Target and binary encoders require the `category-encoders` package:
```bash
pip install lazypredict[encoders]
```

pandas `category`, `string` and (pandas 3) `str` columns are treated as
categorical features alongside `object` columns.

## Intel Extension Acceleration

For improved performance on Intel CPUs, install Intel Extension for Scikit-learn:

```bash
pip install scikit-learn-intelex
```

Patch scikit-learn **before** importing lazypredict, so that the accelerated
estimators are the ones lazypredict collects:

```python
from sklearnex import patch_sklearn
patch_sklearn()

from lazypredict.Supervised import LazyClassifier
```

> **Changed in 0.4.0:** Earlier versions called `patch_sklearn()` automatically on `import lazypredict`, which changed scikit-learn for the whole Python process. The patch is now opt-in.

## MLflow Integration

Lazy Predict includes built-in MLflow integration. Enable it by setting the MLflow tracking URI:

```python
import os
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'

# MLflow tracking will be automatically enabled
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
```

Each model gets its own MLflow run (nested under your active run, if any) with:
- Model metrics (R-squared, RMSE, etc.)
- Training time
- The model name and the estimator's hyperparameters as parameters
- A custom metric, as `custom_metric`
- The fitted pipeline as an artifact (not registered in the model registry)

Lazy Predict does not call `mlflow.autolog()`; enable it yourself if you want
autologging for the rest of your code.