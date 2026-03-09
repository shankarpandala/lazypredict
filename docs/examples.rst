========
Examples
========

This page contains detailed examples of how to use Lazy Predict in various scenarios.

Classification Example
--------------------

Basic Classification
~~~~~~~~~~~~~~~~~~~

Here's a basic example using the breast cancer dataset:

.. code-block:: python

    from lazypredict.Supervised import LazyClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Create classifier
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    
    # Fit and get models
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)

Classification with Custom Metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use custom metrics for model evaluation:

.. code-block:: python

    from sklearn.metrics import f1_score

    def custom_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=custom_f1)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

Advanced Classification Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use advanced options like categorical encoding, timeout, and cross-validation:

.. code-block:: python

    clf = LazyClassifier(
        verbose=1,                          # Show progress
        ignore_warnings=True,               # Suppress warnings
        custom_metric=None,                 # Use default metrics
        predictions=True,                   # Return predictions
        classifiers='all',                  # Use all available classifiers
        categorical_encoder='onehot',       # Encoding strategy
        timeout=60,                         # Max time per model in seconds
        cv=5                                # Cross-validation folds
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

Regression Example
----------------

Basic Regression
~~~~~~~~~~~~~~

Here's an example using the diabetes dataset:

.. code-block:: python

    from lazypredict.Supervised import LazyRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # Load data
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Create and fit regressor
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    print(models)

Working with Pandas DataFrames
---------------------------

Lazy Predict works seamlessly with pandas DataFrames:

.. code-block:: python

    import pandas as pd
    
    # Your DataFrame
    df = pd.DataFrame(X, columns=diabetes.feature_names)
    
    # Split features and target
    X = df
    y = pd.Series(diabetes.target)
    
    # Rest remains the same
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

Categorical Feature Encoding
--------------------------

Lazy Predict supports multiple categorical encoding strategies:

OneHot Encoding (Default)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from lazypredict.Supervised import LazyClassifier
    from sklearn.model_selection import train_test_split

    # Load data with categorical features
    df = pd.read_csv('data_with_categories.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Default onehot encoding
    clf = LazyClassifier(categorical_encoder='onehot', verbose=0)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

Ordinal Encoding
~~~~~~~~~~~~~~

Useful for ordered categorical features or when one-hot encoding creates too many features:

.. code-block:: python

    clf = LazyClassifier(categorical_encoder='ordinal', verbose=0)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

Target Encoding
~~~~~~~~~~~~~

Target encoding requires the ``category-encoders`` package:

.. code-block:: bash

    pip install category-encoders

.. code-block:: python

    clf = LazyClassifier(categorical_encoder='target', verbose=0)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

Binary Encoding
~~~~~~~~~~~~~

Binary encoding is efficient for high-cardinality features:

.. code-block:: python

    # Requires category-encoders package
    clf = LazyClassifier(categorical_encoder='binary', verbose=0)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

Comparing Encoders
~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    
    results = {}
    for encoder in ['onehot', 'ordinal', 'target', 'binary']:
        try:
            clf = LazyClassifier(
                categorical_encoder=encoder,
                verbose=0,
                ignore_warnings=True
            )
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            results[encoder] = models.head(3)
            print(f"\n{encoder.upper()} Encoding Results:")
            print(models.head(3))
        except Exception as e:
            print(f"{encoder}: {e}")

Using with MLflow
---------------

Lazy Predict has built-in MLflow integration for experiment tracking. You can enable it by setting the MLflow tracking URI:

.. code-block:: python

    import os
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'  # Local SQLite tracking
    # Or for remote tracking:
    # os.environ['MLFLOW_TRACKING_URI'] = 'http://your-mlflow-server:5000'

    # MLflow tracking will be automatically enabled
    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

The following metrics and artifacts will be automatically logged to MLflow:

* Model metrics (R-squared, RMSE, etc.)
* Training time
* Model parameters
* Model signatures
* Custom metrics (if provided)
* Model artifacts for each trained model

You can view the results in the MLflow UI:

.. code-block:: bash

    mlflow ui

For Databricks users:
~~~~~~~~~~~~~~~~~~

If you're using Databricks, MLflow tracking is automatically configured:

.. code-block:: python

    # MLflow tracking will use Databricks tracking URI automatically
    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

Getting Model Objects
------------------

You can access the trained model objects:

.. code-block:: python

    # Get all trained models
    model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)

    # Access specific model
    random_forest = model_dictionary['RandomForestRegressor']
    
    # Make predictions with specific model
    predictions = random_forest.predict(X_test)

Model Timeout
------------

Set a maximum time limit for each model to prevent long-running models from blocking:

.. code-block:: python

    # Limit each model to 60 seconds
    clf = LazyClassifier(timeout=60, verbose=1)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    # Models that exceed the timeout will be skipped
    # Check for skipped models in the verbose output

This is particularly useful when:

* Working with large datasets where some models might take very long
* Running experiments with time constraints
* Preventing specific slow models from blocking the entire pipeline

Intel Extension Acceleration
--------------------------

For improved performance on Intel CPUs, install Intel Extension for Scikit-learn:

.. code-block:: bash

    pip install scikit-learn-intelex

Lazy Predict will automatically detect and use it for acceleration:

.. code-block:: python

    # No code changes needed - acceleration is automatic
    clf = LazyClassifier(verbose=0)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    # You'll see "Intel(R) Extension for Scikit-learn enabled" in verbose output

Time Series Forecasting
-----------------------

Basic Forecasting
~~~~~~~~~~~~~~~~

Benchmark 20+ forecasting models with a single call:

.. code-block:: python

    import numpy as np
    from lazypredict.TimeSeriesForecasting import LazyForecaster

    # Generate sample data with trend + seasonality
    np.random.seed(42)
    t = np.arange(200)
    y = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 200)

    y_train, y_test = y[:180], y[180:]

    fcst = LazyForecaster(verbose=0, ignore_warnings=True, predictions=True)
    scores, predictions = fcst.fit(y_train, y_test)
    print(scores)

Forecasting with Seasonal Period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the seasonal period is auto-detected via ACF. Override it manually:

.. code-block:: python

    fcst = LazyForecaster(
        seasonal_period=12,    # monthly data with yearly cycle
        verbose=0,
        ignore_warnings=True,
    )
    scores, predictions = fcst.fit(y_train, y_test)

Forecasting with Exogenous Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass optional exogenous features to models that support them (SARIMAX, AutoARIMA, ML models):

.. code-block:: python

    # Create exogenous features
    X_train = np.column_stack([np.sin(t[:180]), np.cos(t[:180])])
    X_test  = np.column_stack([np.sin(t[180:]), np.cos(t[180:])])

    fcst = LazyForecaster(verbose=0, ignore_warnings=True)
    scores, predictions = fcst.fit(y_train, y_test, X_train, X_test)

Forecasting with Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use expanding-window time series cross-validation:

.. code-block:: python

    fcst = LazyForecaster(
        cv=5,                  # 5-fold TimeSeriesSplit
        verbose=0,
        ignore_warnings=True,
    )
    scores, predictions = fcst.fit(y_train, y_test)
    # scores will contain CV Mean and CV Std columns for each metric

Selecting Specific Forecasters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run only a subset of models:

.. code-block:: python

    fcst = LazyForecaster(
        forecasters=["Holt", "AutoARIMA", "Ridge_TS", "LSTM_TS"],
        verbose=0,
        ignore_warnings=True,
    )
    scores, predictions = fcst.fit(y_train, y_test)

Custom Forecasting Metric
~~~~~~~~~~~~~~~~~~~~~~~~~

Add a custom metric alongside the defaults:

.. code-block:: python

    def median_absolute_error(y_true, y_pred):
        return float(np.median(np.abs(y_true - y_pred)))

    fcst = LazyForecaster(
        custom_metric=median_absolute_error,
        verbose=0,
        ignore_warnings=True,
    )
    scores, predictions = fcst.fit(y_train, y_test)
    # scores will include a 'median_absolute_error' column

Saving and Loading Forecaster Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save all fitted models
    fcst.save_models("./my_forecasters")

    # Load them back
    fcst2 = LazyForecaster()
    fcst2.load_models("./my_forecasters")

    # Use loaded models to forecast
    new_forecasts = fcst2.predict(y_history=y_train, horizon=20)