.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Lazy Predict, run this command in your terminal:

.. code-block:: console

    $ pip install lazypredict

This is the preferred method to install Lazy Predict, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

Optional extras
~~~~~~~~~~~~~~~

Install with time series forecasting support:

.. code-block:: console

    $ pip install lazypredict[timeseries]               # statsmodels + pmdarima
    $ pip install lazypredict[timeseries,deeplearning]   # + LSTM/GRU via PyTorch
    $ pip install lazypredict[timeseries,foundation]     # + Google TimesFM (Python 3.10-3.11 only)

Install with boosting libraries (XGBoost, LightGBM):

.. code-block:: console

    $ pip install lazypredict[boost]

Install with all optional dependencies:

.. code-block:: console

    $ pip install lazypredict[all]

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Lazy Predict can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/shankarpandala/lazypredict

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/shankarpandala/lazypredict/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/shankarpandala/lazypredict
.. _tarball: https://github.com/shankarpandala/lazypredict/tarball/master
