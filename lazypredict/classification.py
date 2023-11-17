from sklearn.base import accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from regression import LazyBaseEstimator
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from utils import adjusted_rsquared

class LazyClassifier(LazyBaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_metrics(self, y_true, y_pred):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred, average="weighted")
        }
        try:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_pred)
        except ValueError:
            metrics["ROC AUC"] = None
        return metrics

    def fit(self, X_train, X_test, y_train, y_test):
        results = []
        for name, model_class in tqdm(self.estimators):
            if "random_state" in model_class().get_params().keys():
                model = model_class(random_state=self.random_state)
            else:
                model = model_class()
            metrics = self._fit_estimator(X_train, y_train, X_test, y_test, model)
            if metrics:
                metrics['Model'] = name
                results.append(metrics)

        self.results_df = pd.DataFrame(results)
        self.results_df.sort_values(by="Balanced Accuracy", ascending=False, inplace=True)
        return self.results_df
