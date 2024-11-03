# lazypredict/explainability/shap_explainer.py

import shap
import pandas as pd  # Added import for pandas
from sklearn.base import BaseEstimator
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class ShapExplainer:
    """
    A class to provide SHAP explainability for machine learning models.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model
        self.explainer = None

    def fit(self, X: pd.DataFrame):
        """
        Fits the SHAP explainer to the model.
        
        Args:
            X (pd.DataFrame): Input data to fit the SHAP explainer.
        """
        try:
            self.explainer = shap.Explainer(self.model, X)
            logger.info("SHAP Explainer has been fitted to the model.")
        except Exception as e:
            logger.error(f"Failed to fit SHAP Explainer: {e}")

    def explain(self, data: pd.DataFrame) -> Optional[Any]:
        """
        Generates SHAP values for the input data.

        Args:
            data (pd.DataFrame): Input data to explain.

        Returns:
            shap_values (Optional[Any]): The SHAP values if explanation is successful, else None.
        """
        try:
            if self.explainer is None:
                raise ValueError("The explainer is not fitted. Call `fit` first.")
            shap_values = self.explainer(data)
            return shap_values
        except Exception as e:
            logger.error(f"Failed to generate SHAP values: {e}")
            return None

    def plot_summary(self, shap_values: Any, feature_names: Optional[list] = None):
        """
        Plots the SHAP summary plot.

        Args:
            shap_values (Any): SHAP values to plot.
            feature_names (Optional[list]): List of feature names.
        """
        try:
            shap.summary_plot(shap_values, feature_names=feature_names)
        except Exception as e:
            logger.error(f"Failed to plot SHAP summary: {e}")
