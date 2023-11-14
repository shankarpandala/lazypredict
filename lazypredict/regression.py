from regression import LazyBaseEstimator
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from utils import adjusted_rsquared

class LazyRegressor(LazyBaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_metrics(self, y_true, y_pred):
        r_squared = r2_score(y_true, y_pred)
        adj_r_squared = adjusted_rsquared(r_squared, len(y_true), len(y_pred))
        return {
            "R-Squared": r_squared,
            "Adjusted R-Squared": adj_r_squared,
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }

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
        self.results_df.sort_values(by="Adjusted R-Squared", ascending=False, inplace=True)
        return self.results_df
