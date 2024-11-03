# lazypredict/explainability/feature_importance.py

import pandas as pd  # Added import for pandas
import numpy as np
from sklearn.base import BaseEstimator
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureImportance:
    """
    A class to compute feature importance for machine learning models.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    def compute_importance(self, X: pd.DataFrame, feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Computes the feature importance of a given model.

        Args:
            X (pd.DataFrame): Input features used for training the model.
            feature_names (Optional[list]): List of feature names, defaults to the columns of X.

        Returns:
            pd.DataFrame: A DataFrame containing feature importances sorted in descending order.
        """
        try:
            # If model has feature_importances_ attribute (e.g., tree-based models)
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = feature_names if feature_names else X.columns.tolist()
            # If model has coef_ attribute (e.g., linear models)
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
                feature_names = feature_names if feature_names else X.columns.tolist()
            else:
                raise ValueError("The model does not have feature_importances_ or coef_ attribute.")

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}")
            return pd.DataFrame()

    def plot_importance(self, importance_df: pd.DataFrame, top_n: int = 10):
        """
        Plots the feature importance.

        Args:
            importance_df (pd.DataFrame): DataFrame containing feature importances.
            top_n (int): Number of top features to plot.
        """
        try:
            import matplotlib.pyplot as plt
            importance_df = importance_df.head(top_n)
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.xlabel('Importance')
            plt.title('Top Feature Importances')
            plt.gca().invert_yaxis()
            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot feature importance: {e}")
