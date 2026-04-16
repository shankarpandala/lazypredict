"""
Regression Example
==================

Benchmark 40+ regressors on the diabetes dataset with two lines of code.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

from lazypredict.Supervised import LazyRegressor

# 1. Load data
diabetes = load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target, random_state=42)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# 2. Fit all regressors
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print("=== Regression Results ===")
print(models)
