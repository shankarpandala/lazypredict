# Lazy Predict

[![image](https://img.shields.io/pypi/v/lazypredict.svg)](https://pypi.python.org/pypi/lazypredict)
[![Publish](https://github.com/shankarpandala/lazypredict/actions/workflows/publish.yml/badge.svg)](https://github.com/shankarpandala/lazypredict/actions/workflows/publish.yml)
[![Documentation Status](https://readthedocs.org/projects/lazypredict/badge/?version=latest)](https://lazypredict.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/lazypredict)](https://pepy.tech/project/lazypredict)
[![CodeFactor](https://www.codefactor.io/repository/github/shankarpandala/lazypredict/badge)](https://www.codefactor.io/repository/github/shankarpandala/lazypredict)

Lazy Predict helps build a lot of basic models without much code and helps understand which models work better without any parameter tuning.

- Free software: MIT license
- Documentation: <https://lazypredict.readthedocs.io>

## Installation

To install Lazy Predict:

```bash
pip install lazypredict
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

| Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
|:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
| LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
| SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
| MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
| Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
| LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
| LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
| SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
| CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
| PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
| LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
| LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
| RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
| GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
| QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
| HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
| RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
| RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
| AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
| ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
| KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
| BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
| BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
| LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
| GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
| NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
| DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
| NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
| ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
| CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
| DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |

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

| Model                         |   Adjusted R-Squared |   R-Squared |     RMSE |   Time Taken |
|:------------------------------|---------------------:|------------:|---------:|-------------:|
| ExtraTreesRegressor           |           0.378921   |  0.520076   |  54.2202 |   0.121466   |
| OrthogonalMatchingPursuitCV   |           0.374947   |  0.517004   |  54.3934 |   0.0111742  |
| Lasso                         |           0.373483   |  0.515873   |  54.457  |   0.00620174 |
| LassoLars                     |           0.373474   |  0.515866   |  54.4575 |   0.0087235  |
| LarsCV                        |           0.3715     |  0.514341   |  54.5432 |   0.0160234  |
| LassoCV                       |           0.370413   |  0.513501   |  54.5903 |   0.0624897  |
| PassiveAggressiveRegressor    |           0.366958   |  0.510831   |  54.7399 |   0.00689793 |
| LassoLarsIC                   |           0.364984   |  0.509306   |  54.8252 |   0.0108321  |
| SGDRegressor                  |           0.364307   |  0.508783   |  54.8544 |   0.0055306  |
| RidgeCV                       |           0.363002   |  0.507774   |  54.9107 |   0.00728202 |
| Ridge                         |           0.363002   |  0.507774   |  54.9107 |   0.00556874 |
| BayesianRidge                 |           0.362296   |  0.507229   |  54.9411 |   0.0122972  |
| LassoLarsCV                   |           0.361749   |  0.506806   |  54.9646 |   0.0175984  |
| TransformedTargetRegressor    |           0.361749   |  0.506806   |  54.9646 |   0.00604773 |
| LinearRegression              |           0.361749   |  0.506806   |  54.9646 |   0.00677514 |
| Lars                          |           0.358828   |  0.504549   |  55.0903 |   0.00935149 |
| ElasticNetCV                  |           0.356159   |  0.502486   |  55.2048 |   0.0478678  |
| HuberRegressor                |           0.355251   |  0.501785   |  55.2437 |   0.0129263  |
| RandomForestRegressor         |           0.349621   |  0.497434   |  55.4844 |   0.2331     |
| AdaBoostRegressor             |           0.340416   |  0.490322   |  55.8757 |   0.0512381  |
| LGBMRegressor                 |           0.339239   |  0.489412   |  55.9255 |   0.0396187  |
| HistGradientBoostingRegressor |           0.335632   |  0.486625   |  56.0779 |   0.0897055  |
| PoissonRegressor              |           0.323033   |  0.476889   |  56.6072 |   0.00953603 |
| ElasticNet                    |           0.301755   |  0.460447   |  57.4899 |   0.00604224 |
| KNeighborsRegressor           |           0.299855   |  0.458979   |  57.5681 |   0.00757337 |
| OrthogonalMatchingPursuit     |           0.292421   |  0.453235   |  57.8729 |   0.00709486 |
| BaggingRegressor              |           0.291213   |  0.452301   |  57.9223 |   0.0302746  |
| GradientBoostingRegressor     |           0.247009   |  0.418143   |  59.7011 |   0.136803   |
| TweedieRegressor              |           0.244215   |  0.415984   |  59.8118 |   0.00633955 |
| XGBRegressor                  |           0.224263   |  0.400567   |  60.5961 |   0.339694   |
| GammaRegressor                |           0.223895   |  0.400283   |  60.6105 |   0.0235181  |
| RANSACRegressor               |           0.203535   |  0.38455    |  61.4004 |   0.0653253  |
| LinearSVR                     |           0.116707   |  0.317455   |  64.6607 |   0.0077076  |
| ExtraTreeRegressor            |           0.00201902 |  0.228833   |  68.7304 |   0.00626636 |
| NuSVR                         |          -0.0667043  |  0.175728   |  71.0575 |   0.0143399  |
| SVR                           |          -0.0964128  |  0.152772   |  72.0402 |   0.0114729  |
| DummyRegressor                |          -0.297553   | -0.00265478 |  78.3701 |   0.00592971 |
| DecisionTreeRegressor         |          -0.470263   | -0.136112   |  83.4229 |   0.00749898 |
| GaussianProcessRegressor      |          -0.769174   | -0.367089   |  91.5109 |   0.0770502  |
| MLPRegressor                  |          -1.86772    | -1.21597    | 116.508  |   0.235267   |
| KernelRidge                   |          -5.03822    | -3.6659     | 169.061  |   0.0243919  |