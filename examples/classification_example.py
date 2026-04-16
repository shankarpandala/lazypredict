"""
Classification Example
======================

Benchmark 30+ classifiers on the breast cancer dataset with two lines of code.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from lazypredict.Supervised import LazyClassifier

# 1. Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

# 2. Fit all classifiers
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print("=== Classification Results ===")
print(models)

# 3. (Optional) Advanced usage with cross-validation and timeout
clf_cv = LazyClassifier(
    verbose=1,
    ignore_warnings=True,
    cv=5,
    timeout=60,
)
models_cv, _ = clf_cv.fit(X_train, X_test, y_train, y_test)

print("\n=== With 5-Fold Cross-Validation ===")
print(models_cv)
