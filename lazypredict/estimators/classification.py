# estimators/classification.py

from .base import BaseEstimator
from ..utils.models import ModelFetcher
from ..metrics.classification_metrics import ClassificationMetrics
from ..explainability.shap_explainer import SHAPExplainer
from ..explainability.lime_explainer import LIMEExplainer
from ..mlflow_integration.mlflow_logger import MLflowLogger
import pandas as pd
import time

class LazyClassifier(BaseEstimator):
    """
    LazyClassifier trains and evaluates multiple classification models, using modular
    components for preprocessing, metrics, explainability, and experiment tracking.

    This class inherits from BaseEstimator and implements `fit`, `predict`, and `evaluate`.
    """

    def __init__(self, verbose=0, ignore_warnings=True, custom_metric=None, predictions=False,
                 random_state=42, explainability=False, mlflow_tracking=False):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.random_state = random_state
        self.explainability = explainability
        self.mlflow_tracking = mlflow_tracking
        self.models = {}
        self.mlflow_logger = MLflowLogger(experiment_name="LazyClassifier Experiment") if self.mlflow_tracking else None

    def fit(self, X, y):
        """
        Trains each model on the provided training data.

        Parameters
        ----------
        X : DataFrame
            Training data features.
        y : Series
            Training data labels.
        """
        self._initialize_preprocessor(X)
        X = self._get_transformed_data(X)
        self.models = {}

        classifiers = ModelFetcher.fetch_classifiers()
        for name, model_class in classifiers:
            try:
                model = model_class(random_state=self.random_state) if 'random_state' in model_class().get_params() else model_class()
                model.fit(X, y)
                self.models[name] = model
                if self.verbose:
                    print(f"Trained {name}")
            except Exception as e:
                if not self.ignore_warnings:
                    print(f"{name} failed: {e}")

    def predict(self, X):
        """
        Generates predictions for each model on the provided data.

        Parameters
        ----------
        X : DataFrame
            Test data features.

        Returns
        -------
        dict
            A dictionary with model names as keys and predictions as values.
        """
        X = self._get_transformed_data(X)
        predictions = {name: model.predict(X) for name, model in self.models.items()}
        return predictions

    def evaluate(self, X, y):
        """
        Evaluates each model on the provided test data and generates a table of performance metrics.

        Parameters
        ----------
        X : DataFrame
            Test data features.
        y : Series
            Test data labels.

        Returns
        -------
        DataFrame
            A DataFrame containing performance metrics for each model.
        dict, optional
            A dictionary containing explanations for each model if `self.explainability` is True.
        """
        X = self._get_transformed_data(X)
        scores = []
        explanations = {}

        metrics_calculator = ClassificationMetrics()

        for name, model in self.models.items():
            start = time.time()
            y_pred = model.predict(X)
            metrics = metrics_calculator.compute(y, y_pred)

            if self.mlflow_tracking:
                self.mlflow_logger.start_run(run_name=name)
                self.mlflow_logger.log_metrics(metrics)
                self.mlflow_logger.log_model(model, model_name=name)
                self.mlflow_logger.end_run()

            scores.append({"Model": name, **metrics, "Time Taken": time.time() - start})

            if self.explainability:
                shap_explainer = SHAPExplainer(model, data=X)
                lime_explainer = LIMEExplainer(model, X)
                explanations[name] = {
                    "SHAP": shap_explainer.explain(X),
                    "LIME": lime_explainer.explain_instance(X, instance=0)
                }

        scores_df = pd.DataFrame(scores).sort_values(by="Balanced Accuracy", ascending=False).set_index("Model")
        return (scores_df, explanations) if self.explainability else scores_df
