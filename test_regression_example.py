"""
Test the regression example from README.md
"""
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

print("=" * 80)
print("Testing Regression Example from README")
print("=" * 80)

# Load data
print("\n1. Loading diabetes dataset...")
diabetes = datasets.load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
X = X.astype(np.float32)
print(f"   Dataset shape: {X.shape}, Target shape: {y.shape}")

# Split data
print("\n2. Splitting data (90/10 split)...")
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Train models
print("\n3. Training multiple regressors...")
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None, predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Display results
print("\n4. Results:")
print("=" * 80)
print(models.head(10))
print("\n" + "=" * 80)
print(f"\nTotal models trained: {len(models)}")
print(f"Best model: {models.index[0]}")
print(f"Best R-Squared: {models.iloc[0]['R-Squared']:.4f}")
print(f"Best RMSE: {models.iloc[0]['RMSE']:.4f}")
print(f"\nPredictions shape: {predictions.shape}")

print("\n✅ Regression example completed successfully!")
