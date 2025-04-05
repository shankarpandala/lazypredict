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