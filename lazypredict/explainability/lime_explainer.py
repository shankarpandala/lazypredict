# lazypredict/explainability/lime_explainer.py

import lime
import lime.lime_tabular
import pandas as pd  # Added import for pandas
from sklearn.base import BaseEstimator
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class LimeExplainer:
    """
    A class to provide LIME explainability for machine learning models.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model
        self.explainer = None

    def fit(self, X: pd.DataFrame):
        """
        Fits the LIME explainer to the model.
        
        Args:
            X (pd.DataFrame): Input data to fit the LIME explainer.
        """
        try:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns.tolist(),
                class_names=['target'],  # Adjust as needed based on the classification/regression problem
                discretize_continuous=True
            )
            logger.info("LIME Explainer has been initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LIME Explainer: {e}")

    def explain(self, data: pd.DataFrame, instance_idx: int) -> Optional[Any]:
        """
        Generates LIME explanation for a specific instance.

        Args:
            data (pd.DataFrame): Input data to explain.
            instance_idx (int): Index of the instance to explain.

        Returns:
            exp (Optional[Any]): LIME explanation object if successful, else None.
        """
        try:
            if self.explainer is None:
                raise ValueError("The explainer is not initialized. Call `fit` first.")
            
            instance = data.iloc[instance_idx]
            exp = self.explainer.explain_instance(
                instance.values, 
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=min(10, len(data.columns))  # Limit number of features for explanation
            )
            return exp
        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {e}")
            return None

    def show_explanation(self, explanation: Any):
        """
        Displays the LIME explanation.

        Args:
            explanation (Any): LIME explanation object to display.
        """
        try:
            explanation.show_in_notebook(show_table=True, show_all=False)
        except Exception as e:
            logger.error(f"Failed to display LIME explanation: {e}")
