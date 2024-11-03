# lazypredict/mlflow_integration/mlflow_logger.py

import mlflow
import pandas as pd  # Added import for pandas
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class MLflowLogger:
    """
    A class for managing MLflow logging operations.
    """

    def __init__(self, experiment_name: str = "Default Experiment"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_params(self, params: dict):
        """
        Logs parameters to the active MLflow run.
        """
        try:
            mlflow.log_params(params)
            logger.info("Logged parameters to MLflow.")
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: dict):
        """
        Logs metrics to the active MLflow run.
        """
        try:
            mlflow.log_metrics(metrics)
            logger.info("Logged metrics to MLflow.")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_model(self, model: Any, artifact_path: str, X: Optional[pd.DataFrame] = None):
        """
        Logs a machine learning model to MLflow.

        Args:
            model (Any): The model to log.
            artifact_path (str): The artifact path where the model will be saved.
            X (Optional[pd.DataFrame]): Optional input data to infer model schema.
        """
        try:
            if X is not None:
                signature = mlflow.models.infer_signature(X, model.predict(X))
                mlflow.sklearn.log_model(model, artifact_path=artifact_path, signature=signature)
            else:
                mlflow.sklearn.log_model(model, artifact_path=artifact_path)

            logger.info(f"Logged model to MLflow at {artifact_path}.")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
