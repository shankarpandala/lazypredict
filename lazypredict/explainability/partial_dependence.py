# lazypredict/explainability/partial_dependence.py

import pandas as pd  # Added import for pandas
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import BaseEstimator
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

class PartialDependence:
    """
    A class to generate partial dependence plots for machine learning models.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    def plot_partial_dependence(self, X: pd.DataFrame, features: List[int], target: Any = None):
        """
        Plots partial dependence for the given model and dataset.

        Args:
            X (pd.DataFrame): Input feature data.
            features (List[int]): List of feature indices or names for which to plot partial dependence.
            target (Any, optional): Target class to plot for classification models.
        """
        try:
            if target:
                PartialDependenceDisplay.from_estimator(self.model, X, features, target=target)
            else:
                PartialDependenceDisplay.from_estimator(self.model, X, features)

            plt.show()

        except Exception as e:
            logger.error(f"Failed to generate partial dependence plot: {e}")
