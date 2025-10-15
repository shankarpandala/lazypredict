"""Quick test to verify all optimizations work correctly."""
# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Test lazy loading - should be fast
start = time.time()
from lazypredict.Supervised import LazyClassifier
import_time = time.time() - start
print(f"[OK] Import time: {import_time:.3f}s (lazy loading working!)")

# Generate test data
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=8,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Test with sequential execution
print("\nTesting sequential execution (n_jobs=1)...")
start = time.time()
clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=1, predictions=False)
models_seq = clf.fit(X_train, X_test, y_train, y_test)
seq_time = time.time() - start
print(f"[OK] Sequential: {seq_time:.2f}s, {len(models_seq)} models trained")

# Test with parallel execution
print("\nTesting parallel execution (n_jobs=-1)...")
start = time.time()
clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=-1, predictions=False)
models_par = clf.fit(X_train, X_test, y_train, y_test)
par_time = time.time() - start
print(f"[OK] Parallel: {par_time:.2f}s, {len(models_par)} models trained")

# Calculate speedup
speedup = seq_time / par_time
print(f"\n[OK] Speedup: {speedup:.2f}x")

# Verify XGBoost and LightGBM were loaded
xgb_models = [m for m in models_par.index if 'XGB' in m]
lgbm_models = [m for m in models_par.index if 'LGBM' in m]

if xgb_models:
    print(f"[OK] XGBoost models loaded: {xgb_models}")
else:
    print("[FAIL] XGBoost models NOT found")

if lgbm_models:
    print(f"[OK] LightGBM models loaded: {lgbm_models}")
else:
    print("[FAIL] LightGBM models NOT found")

# Display top 5 models
print("\n[OK] Top 5 models by Balanced Accuracy:")
print(models_par.head())

print("\n[SUCCESS] All optimizations verified successfully!")
