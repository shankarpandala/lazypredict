"""
Test the classification example from README.md
"""
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

print("=" * 80)
print("Testing Classification Example from README")
print("=" * 80)

# Load data
print("\n1. Loading breast cancer dataset...")
data = load_breast_cancer()
X = data.data
y = data.target
print(f"   Dataset shape: {X.shape}, Target shape: {y.shape}")

# Split data
print("\n2. Splitting data (50/50 split)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Train models
print("\n3. Training multiple classifiers...")
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display results
print("\n4. Results:")
print("=" * 80)
print(models.head(10))
print("\n" + "=" * 80)
print(f"\nTotal models trained: {len(models)}")
print(f"Best model: {models.index[0]}")
print(f"Best accuracy: {models.iloc[0]['Accuracy']:.4f}")
print(f"\nPredictions shape: {predictions.shape}")

print("\n✅ Classification example completed successfully!")
