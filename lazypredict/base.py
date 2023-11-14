from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import numeric_transformer, categorical_transformer_low, categorical_transformer_high

# Define the BaseLazyEstimator class
class LazyBaseEstimator:
    def __init__(self, verbose=0, ignore_warnings=True, custom_metric=None, predictions=False, random_state=42, estimators="all"):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.estimators = estimators

    def _create_pipeline(self, estimator):
        # Assuming numeric_transformer, categorical_transformer_low, and categorical_transformer_high are defined elsewhere
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        steps = [("preprocessor", preprocessor), ("estimator", estimator)]
        return Pipeline(steps=steps)

    def _fit_estimator(self, X_train, y_train, X_test, y_test, estimator):
        start = time.time()
        try:
            estimator = self._create_pipeline(estimator)
            estimator.fit(X_train, y_train)
            self.models[estimator_name] = estimator
            y_pred = estimator.predict(X_test)

            metrics = self._calculate_metrics(y_test, y_pred)
            metrics['Time Taken'] = time.time() - start
            return metrics
        except Exception as e:
            if not self.ignore_warnings:
                print(f"{estimator_name} model failed to execute: {e}")

    def _calculate_metrics(self, y_true, y_pred):
        # Override this method in derived classes
        pass

    def fit(self, X_train, X_test, y_train, y_test):
        # Implementation in derived classes
        pass
