# Installation

## Stable release

To install Lazy Predict, run this command in your terminal:

```console
pip install lazypredict
```

This is the preferred method to install Lazy Predict, as it will always install
the most recent stable release.

If you don't have [pip][pip] installed, this [Python installation guide][pyguide]
can guide you through the process.

### Optional extras

Install with boosting libraries (XGBoost, LightGBM, CatBoost):

```console
pip install lazypredict[boost]
```

Install with time series forecasting support:

```console
pip install lazypredict[timeseries]                  # statsmodels + pmdarima
pip install lazypredict[timeseries,deeplearning]     # + LSTM/GRU via PyTorch
pip install lazypredict[timeseries,foundation]       # + Google TimesFM (Python 3.10-3.11 only)
```

Install with all optional dependencies:

```console
pip install lazypredict[all]
```

### GPU acceleration extras

For GPU-accelerated model training:

```console
pip install lazypredict[boost]    # XGBoost, LightGBM, CatBoost (all support GPU)
pip install cuml-cu12             # cuML (RAPIDS) GPU-native sklearn models
```

Then enable GPU in your code with `use_gpu=True`. See the
[examples](examples.md) page for details.

## From sources

The sources for Lazy Predict can be downloaded from the [Github repo][repo].

You can either clone the public repository:

```console
git clone git://github.com/shankarpandala/lazypredict
```

Or download the [tarball][tarball]:

```console
curl -OJL https://github.com/shankarpandala/lazypredict/tarball/master
```

Once you have a copy of the source, you can install it with:

```console
pip install .
```

[pip]: https://pip.pypa.io
[pyguide]: http://docs.python-guide.org/en/latest/starting/installation/
[repo]: https://github.com/shankarpandala/lazypredict
[tarball]: https://github.com/shankarpandala/lazypredict/tarball/master
