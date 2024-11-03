from lime.lime_tabular import LimeTabularExplainer

class LIMEExplainer:
    """
    LIMEExplainer for interpreting model predictions using LIME.

    This class provides methods for explaining individual predictions using
    LIME explanations.

    Attributes
    ----------
    model : object
        The model to be explained.
    explainer : LimeTabularExplainer
        LIME explainer instance.

    Methods
    -------
    explain_instance(X, instance):
        Explain a single instance using LIME.
    """

    def __init__(self, model, training_data):
        """
        Parameters
        ----------
        model : object
            A trained model compatible with LIME explainers.
        training_data : DataFrame
            Training data used for creating the explainer.
        """
        self.model = model
        self.explainer = LimeTabularExplainer(
            training_data.values,
            feature_names=training_data.columns,
            mode="classification" if hasattr(model, "predict_proba") else "regression"
        )

    def explain_instance(self, X, instance):
        """
        Explain a single instance using LIME.

        Parameters
        ----------
        X : DataFrame
            Input data containing the instance to explain.
        instance : int
            Index of the instance to explain in the data.

        Returns
        -------
        dict
            Explanation results with feature importance for the instance.
        """
        explanation = self.explainer.explain_instance(
            X.iloc[instance].values, self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
        )
        return explanation.as_list()
