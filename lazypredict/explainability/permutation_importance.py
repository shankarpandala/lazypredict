# lazypredict/explainability/permutation_importance.py

import pandas as pd  # Added import for pandas
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from typing import Any
import logging

logger = logging.getLogger(__name__)

class PermutationImportance:
    """
    A class to compute permutation importance for machine learning models.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    def compute_importance(self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
        """
        Computes permutation importance for the given model and dataset.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            n_repeats (int): The number of times to permute a feature.
            random_state (int): The seed for random permutation.

        Returns:
            pd.DataFrame: A DataFrame containing the permutation importances sorted in descending order.
        """
        try:
            result = permutation_importance(self.model, X, y, n_repeats=n_repeats, random_state=random_state)
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance Mean': result.importances_mean,
                'Importance Std': result.importances_std
            }).sort_values(by='Importance Mean', ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"Failed to compute permutation importance: {e}")
            return pd.DataFrame()

    def plot_importance(self, importance_df: pd.DataFrame, top_n: int = 10):
        """
        Plots the permutation importance.

        Args:
            importance_df (pd.DataFrame): DataFrame containing permutation importances.
            top_n (int): Number of top features to plot.
        """
        try:
            import matplotlib.pyplot as plt
            importance_df = importance_df.head(top_n)
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance Mean'])
            plt.xlabel('Importance Mean')
            plt.title('Top Permutation Importances')
            plt.gca().invert_yaxis()
            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot permutation importance: {e}")
