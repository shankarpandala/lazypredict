"""Debug script to check multi-class SHAP shapes."""

from lazypredict.Explainer import ModelExplainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import numpy as np

# Load data
X, y = load_iris(return_X_y=True)
print(f'X shape: {X.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'X_train shape: {X_train.shape}')

# Train model
clf = LazyClassifier(verbose=0, ignore_warnings=True, classifiers=['LogisticRegression'])
models = clf.fit(X_train, X_test, y_train, y_test)

# Create explainer
explainer = ModelExplainer(clf, X_train, X_test)
print(f'Feature names count: {len(explainer.feature_names)}')
print(f'Feature names: {explainer.feature_names}')

# Compute SHAP values
shap_values = explainer._compute_shap_values('LogisticRegression')
print(f'SHAP values shape: {shap_values.shape}')

# Apply mean
importance_scores = np.abs(shap_values).mean(axis=0)
print(f'After mean(axis=0): {importance_scores.shape}')

# Flatten if needed (average across classes, not features)
if importance_scores.ndim > 1:
    importance_scores = importance_scores.mean(axis=1)
    print(f'After flattening: {importance_scores.shape}')

print(f'\nFinal importance_scores length: {len(importance_scores)}')
print(f'Feature names length: {len(explainer.feature_names)}')
