================
Explainability
================

LazyPredict now includes comprehensive model explainability features using SHAP (SHapley Additive exPlanations). This allows you to understand which features are most important for your models' predictions.

Installation
------------

To use explainability features, you need to install SHAP:

.. code-block:: bash

    pip install shap

Or install LazyPredict with explainability support:

.. code-block:: bash

    pip install lazypredict[explainability]

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from lazypredict.Supervised import LazyClassifier
    from lazypredict.Explainer import ModelExplainer
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load and split data
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train models
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models = clf.fit(X_train, X_test, y_train, y_test)

    # Create explainer
    explainer = ModelExplainer(clf, X_train, X_test)

    # Get feature importance
    importance = explainer.feature_importance('LogisticRegression', top_n=10)
    print(importance)

Features
--------

1. Feature Importance
~~~~~~~~~~~~~~~~~~~~~

Get ranked feature importance for any trained model:

.. code-block:: python

    # Get top 10 features
    importance = explainer.feature_importance('RandomForestClassifier', top_n=10)
    
    # Get all features
    importance = explainer.feature_importance('LogisticRegression')
    
    # Use signed importance (positive/negative impact)
    importance = explainer.feature_importance('LinearRegression', absolute=False)

2. Summary Plots
~~~~~~~~~~~~~~~~

Visualize feature importance across all samples:

.. code-block:: python

    # Dot plot (default)
    explainer.plot_summary('LogisticRegression', plot_type='dot')
    
    # Bar plot for overall importance
    explainer.plot_summary('RandomForestClassifier', plot_type='bar')
    
    # Violin plot for distribution
    explainer.plot_summary('GradientBoostingClassifier', plot_type='violin')

3. Single Prediction Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understand individual predictions with waterfall plots:

.. code-block:: python

    # Explain first test sample
    explainer.explain_prediction('LogisticRegression', instance_idx=0)
    
    # Explain 10th sample
    explainer.explain_prediction('RandomForestClassifier', instance_idx=9)

4. Dependence Plots
~~~~~~~~~~~~~~~~~~~

Show how a feature affects predictions:

.. code-block:: python

    # For regression
    explainer.plot_dependence('Ridge', feature='age')
    
    # For classification
    explainer.plot_dependence('LogisticRegression', feature='mean radius')

5. Model Comparison
~~~~~~~~~~~~~~~~~~~

Compare feature importance across different models:

.. code-block:: python

    # Compare top features
    comparison = explainer.compare_models(
        ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier'],
        top_n_features=10
    )
    print(comparison)

6. Top Features for Specific Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the most influential features for a specific prediction:

.. code-block:: python

    top_features = explainer.get_top_features('LogisticRegression', instance_idx=0, top_n=5)
    print(top_features)

Complete Example
----------------

Classification with Explainability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lazypredict.Supervised import LazyClassifier
    from lazypredict.Explainer import ModelExplainer
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    clf = LazyClassifier(verbose=0, ignore_warnings=True, 
                        classifiers=['LogisticRegression', 'RandomForestClassifier'])
    models = clf.fit(X_train, X_test, y_train, y_test)

    print("Best model:", models.index[0])

    # Create explainer
    explainer = ModelExplainer(clf, X_train, X_test)

    # 1. Get feature importance
    print("\\nTop 10 Most Important Features:")
    importance = explainer.feature_importance(models.index[0], top_n=10)
    print(importance)

    # 2. Generate summary plot
    print("\\nGenerating SHAP summary plot...")
    explainer.plot_summary(models.index[0], show=True)

    # 3. Explain a specific prediction
    print("\\nExplaining prediction for first test sample...")
    explainer.explain_prediction(models.index[0], instance_idx=0, show=True)

    # 4. Compare models
    print("\\nComparing feature importance across models:")
    comparison = explainer.compare_models(
        [models.index[0], models.index[1]],
        top_n_features=5
    )
    print(comparison)

Regression with Explainability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lazypredict.Supervised import LazyRegressor
    from lazypredict.Explainer import ModelExplainer
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train models
    reg = LazyRegressor(verbose=0, ignore_warnings=True,
                       regressors=['Ridge', 'RandomForestRegressor'])
    models = reg.fit(X_train, X_test, y_train, y_test)

    # Create explainer
    explainer = ModelExplainer(reg, X_train, X_test)

    # Get feature importance
    importance = explainer.feature_importance('Ridge')
    print(importance)

    # Plot dependence for top feature
    top_feature = importance.iloc[0]['feature']
    explainer.plot_dependence('Ridge', feature=top_feature)

Multi-Class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from lazypredict.Supervised import LazyClassifier
    from lazypredict.Explainer import ModelExplainer
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train models
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models = clf.fit(X_train, X_test, y_train, y_test)

    # Create explainer
    explainer = ModelExplainer(clf, X_train, X_test)

    # Feature importance (averaged across all classes)
    importance = explainer.feature_importance('LogisticRegression', top_n=4)
    print(importance)

Supported Models
----------------

The ModelExplainer automatically selects the appropriate SHAP explainer based on the model type:

**Tree-based models** (TreeExplainer):
- RandomForestClassifier/Regressor
- ExtraTreesClassifier/Regressor
- GradientBoostingClassifier/Regressor
- XGBClassifier/Regressor
- LGBMClassifier/Regressor
- CatBoostClassifier/Regressor
- DecisionTreeClassifier/Regressor

**Linear models** (LinearExplainer):
- LinearRegression
- LogisticRegression
- Ridge/RidgeClassifier
- Lasso
- ElasticNet
- SGDClassifier/Regressor

**Other models** (KernelExplainer):
- SVC/SVR
- KNeighborsClassifier/Regressor
- MLPClassifier/Regressor
- GaussianNB
- And all other sklearn models

Performance Tips
----------------

1. **Use sampling for large datasets:**

.. code-block:: python

    # Sample background data for faster computation
    from sklearn.utils import resample
    X_train_sample = resample(X_train, n_samples=100, random_state=42)
    explainer = ModelExplainer(clf, X_train_sample, X_test)

2. **Save and reuse SHAP values:**

.. code-block:: python

    # Compute once
    explainer.feature_importance('LogisticRegression')
    
    # Use multiple times (SHAP values are cached)
    explainer.plot_summary('LogisticRegression')
    explainer.explain_prediction('LogisticRegression', instance_idx=0)

3. **Limit displayed features:**

.. code-block:: python

    # Show only top features
    explainer.plot_summary('RandomForestClassifier', max_display=10)

Troubleshooting
---------------

**ImportError: SHAP not available**

Install SHAP:

.. code-block:: bash

    pip install shap

**Slow computation for large datasets**

Use a smaller background sample:

.. code-block:: python

    X_train_sample = X_train.sample(n=100)
    explainer = ModelExplainer(clf, X_train_sample, X_test)

**Model not found error**

Ensure the model name matches exactly:

.. code-block:: python

    # Check available models
    print(clf.trained_models.keys())
    
    # Use exact name
    explainer.feature_importance('LogisticRegression')

For more details, see the :doc:`modules` API reference or the `GitHub repository <https://github.com/shankarpandala/lazypredict>`_.
