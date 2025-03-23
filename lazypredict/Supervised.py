"""Legacy Supervised module for backward compatibility."""
from .models.classification import LazyClassifier
from .models.regression import LazyRegressor

# Keep these for backward compatibility
Supervised = LazyClassifier
LazyClassification = LazyClassifier
LazyRegression = LazyRegressor
