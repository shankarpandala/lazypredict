# Installation

## Stable release

Install from PyPI (recommended):

```console
pip install lazypredict
```

Install from conda-forge:

```console
conda install -c conda-forge lazypredict
```

### Core dependencies

The base install includes: `scikit-learn`, `pandas`, `numpy`, `tqdm`, `joblib`, `click`.

---

## Optional extras

Lazy Predict has several optional dependency groups. Install them with pip extras:

### Boosting libraries

XGBoost, LightGBM, and CatBoost (all support GPU acceleration):

```console
pip install lazypredict[boost]
```

On macOS, XGBoost and LightGBM also need the OpenMP runtime:

```console
brew install libomp
```

Without it, lazypredict warns that the library could not be imported and skips its models.

### Categorical encoders

Target and binary encoding (`categorical_encoder="target"` / `"binary"`) via
`category-encoders`:

```console
pip install lazypredict[encoders]
```

### Time series forecasting

Statistical models (Holt-Winters, SARIMAX, Theta):

```console
pip install lazypredict[timeseries]
```

Add deep learning models (LSTM, GRU via PyTorch):

```console
pip install lazypredict[timeseries,deeplearning]
```

Add the Google TimesFM foundation model (installs `timesfm[torch]`):

```console
pip install lazypredict[timeseries,foundation]
```

### MLflow experiment tracking

```console
pip install lazypredict[mlflow]
```

Activate tracking by setting an environment variable before running:

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
```

### Hyperparameter tuning

```console
pip install lazypredict[tune]      # Optuna backend
pip install lazypredict[flaml]     # FLAML backend
```

Enable with `tune=True` in any Lazy estimator:

```python
clf = LazyClassifier(tune=True, tune_top_k=5, tune_trials=50)
```

### Visualization

```console
pip install lazypredict[viz]
```

Required for `LazyForecaster.plot_results()` and all plotting utilities.

### Explainability

```console
pip install lazypredict[explain]     # SHAP values
pip install lazypredict[interpret]   # InterpretML (EBM)
```

### Distributed / Spark

```console
pip install lazypredict[spark]       # PySpark MLlib
```

### All optional dependencies

```console
pip install lazypredict[all]
```

---

## GPU acceleration

GPU acceleration requires an NVIDIA GPU with CUDA support:

```console
pip install lazypredict[boost]   # XGBoost, LightGBM, CatBoost (all support GPU)
pip install cuml-cu12            # cuML (RAPIDS) GPU-native sklearn replacements
```

Enable with `use_gpu=True`:

```python
clf = LazyClassifier(use_gpu=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

Supported GPU backends:

| Backend | Package | Parameter |
|---------|---------|-----------|
| XGBoost | `xgboost` | `device="cuda"` |
| LightGBM | `lightgbm` | `device="gpu"` |
| CatBoost | `catboost` | `task_type="GPU"` |
| cuML | `cuml-cu12` | GPU-native (no extra params) |
| LSTM / GRU | `torch` with CUDA | CUDA device |
| TimesFM | `timesfm` | CUDA device |

All GPU backends fall back to CPU automatically if CUDA is not available.

---

## Intel CPU acceleration

For improved performance on Intel CPUs:

```console
pip install scikit-learn-intelex
```

Patch scikit-learn **before** importing lazypredict, so that the accelerated
estimators are the ones lazypredict collects:

```python
from sklearnex import patch_sklearn
patch_sklearn()

from lazypredict.Supervised import LazyClassifier
```

!!! note "Changed in 0.4.0"
    Earlier versions called `patch_sklearn()` automatically on `import lazypredict`,
    which changed scikit-learn for the whole Python process. The patch is now opt-in.

---

## Development install

To set up a local development environment:

```console
git clone https://github.com/shankarpandala/lazypredict
cd lazypredict
pip install -e ".[all,dev]"
```

Run the tests:

```console
pytest tests/
```

---

## Installing from source (archive)

Download and install without git:

```console
curl -OJL https://github.com/shankarpandala/lazypredict/tarball/master
pip install lazypredict-*.tar.gz
```

[pip]: https://pip.pypa.io
[pyguide]: http://docs.python-guide.org/en/latest/starting/installation/
