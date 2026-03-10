=====
Usage
=====

To use Lazy Predict in a project::

    import lazypredict

==============
Classification
==============

Example ::

    from lazypredict.Supervised import LazyClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    models

==========
Regression
==========

Example ::

    from lazypredict.Supervised import LazyRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.utils import shuffle
    import numpy as np
    diabetes = load_diabetes()
    X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
    models,predictions = reg.fit(X_train, X_test, y_train, y_test)


.. warning::
    Regression and Classification are replaced with LazyRegressor and LazyClassifier.
    Regression and Classification classes will be removed in next release

========================
Time Series Forecasting
========================

LazyForecaster benchmarks 20+ statistical, machine-learning, deep-learning,
and pretrained foundation models on your time series with a single call.

Example ::

    import numpy as np
    from lazypredict.TimeSeriesForecasting import LazyForecaster

    # Generate sample data
    np.random.seed(42)
    t = np.arange(200)
    y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 200)

    y_train, y_test = y[:180], y[180:]

    fcst = LazyForecaster(verbose=0, ignore_warnings=True)
    scores, predictions = fcst.fit(y_train, y_test)
    print(scores)

With exogenous variables ::

    X_train = np.column_stack([np.sin(t[:180]), np.cos(t[:180])])
    X_test  = np.column_stack([np.sin(t[180:]), np.cos(t[180:])])

    fcst = LazyForecaster(verbose=0, ignore_warnings=True)
    scores, predictions = fcst.fit(y_train, y_test, X_train, X_test)

With cross-validation and custom options ::

    fcst = LazyForecaster(
        verbose=1,
        seasonal_period=12,   # override auto-detection
        cv=3,                 # 3-fold TimeSeriesSplit CV
        timeout=30,           # max 30 seconds per model
        sort_by="MAE",
    )
    scores, predictions = fcst.fit(y_train, y_test)

==================
GPU Acceleration
==================

Enable GPU acceleration for supported models with ``use_gpu=True``:

.. code-block:: python

    clf = LazyClassifier(use_gpu=True, verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    reg = LazyRegressor(use_gpu=True, verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    fcst = LazyForecaster(use_gpu=True, verbose=0, ignore_warnings=True)
    scores, predictions = fcst.fit(y_train, y_test)

GPU-accelerated models:

* **XGBoost** (``device="cuda"``)
* **LightGBM** (``device="gpu"``)
* **CatBoost** (``task_type="GPU"``)
* **cuML (RAPIDS)** --- GPU-native sklearn replacements (auto-discovered)
* **LSTM / GRU** --- PyTorch CUDA
* **TimesFM** --- PyTorch CUDA

Falls back to CPU automatically if no CUDA GPU is available.
