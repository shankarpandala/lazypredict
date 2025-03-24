.. lazypredict documentation master file, created by
   sphinx-quickstart on Mon Mar 24 13:22:55 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lazypredict's documentation!
=======================================

Lazy Predict helps build a lot of basic models without much code and helps understand 
which models work better without any parameter tuning.

Installation
============

To install Lazy Predict:

.. code-block:: bash

    pip install lazypredict

Usage
-----

To use Lazy Predict in a project:

.. code-block:: python

    import lazypredict
    from lazypredict.Supervised import LazyClassifier, LazyRegressor
    
    # For classification
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    # For regression  
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

