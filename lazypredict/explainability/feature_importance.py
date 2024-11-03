import pandas as pd

class FeatureImportance:
    """
    FeatureImportance for extracting feature importances from a trained model.

    This class provides methods to obtain and display feature importance values
    for models with a built-in feature_importances_ attribute.

    Attributes
    ----------
    model : object
        The model with a feature_importances_ attribute.

    Methods
    -------
    get_importance():
        Returns a DataFrame of feature importances.
    plot_importance():
        Plots the feature importances.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : object
            A trained model with a feature_importances_ attribute.
        """
        self.model = model

    def get_importance(self):
        """
        Returns a DataFrame of feature importances.

        Returns
        -------
        DataFrame
            Feature importances for each feature.
        """
        return pd.DataFrame({
            "feature": range(len(self.model.feature_importances_)),
            "importance": self.model.feature_importances_
        })

    def plot_importance(self):
        """
        Plot the feature importances.
        """
        import matplotlib.pyplot as plt

        importance_df = self.get_importance().sort_values(by="importance", ascending=False)
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df["feature"].astype(str), importance_df["importance"])
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.show()
