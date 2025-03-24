"""
MLflow utilities for lazypredict.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("lazypredict.mlflow")

# Flag to track if MLflow is available
MLFLOW_AVAILABLE = False
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    logger.warning("MLflow not installed. Experiment tracking disabled.")


def configure_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow tracking.

    Parameters
    ----------
    tracking_uri : str, optional (default=None)
        MLflow tracking URI. If None, will use the global tracking URI.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Set the tracking URI
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        elif "MLFLOW_TRACKING_URI" in os.environ:
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    except Exception as e:
        logger.error(f"Error configuring MLflow: {e}")


def start_run(
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = "lazypredict",
    tags: Optional[Dict[str, str]] = None,
) -> Optional[Any]:
    """Start a new MLflow run.

    Parameters
    ----------
    run_name : str, optional (default=None)
        Name of the run.
    experiment_name : str, optional (default="lazypredict")
        Name of the experiment.
    tags : Dict[str, str], optional (default=None)
        Tags to set on the run.

    Returns
    -------
    Optional[Any]
        The active MLflow run, if successful.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return None

    try:
        # End any active run safely
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        # Set experiment
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(
                f"Could not set experiment {experiment_name}: {e}. Using default experiment."
            )

        # Start run
        run = mlflow.start_run(run_name=run_name)

        # Set tags
        if tags and run:
            mlflow.set_tags(tags)

        if run and run.info:
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run
        else:
            logger.warning("Failed to start MLflow run")
            return None
    except Exception as e:
        logger.error(f"Error starting MLflow run: {e}")
        return None


def end_run() -> None:
    """End the active MLflow run."""
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        active_run = mlflow.active_run()
        if active_run and active_run.info:
            run_id = active_run.info.run_id
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {run_id}")
    except Exception as e:
        logger.error(f"Error ending MLflow run: {e}")


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameters to log.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Ensure we have an active run
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run. Starting a new run.")
            start_run()

        # Clean params to ensure all values are MLflow compatible
        clean_params = {}
        for k, v in params.items():
            if v is None:
                clean_params[k] = "None"
            elif isinstance(v, (str, int, float, bool)):
                clean_params[k] = v
            else:
                clean_params[k] = str(v)

        mlflow.log_params(clean_params)
        logger.debug(f"Logged parameters: {clean_params}")
    except Exception as e:
        logger.error(f"Error logging parameters: {e}")


def log_metric(key: str, value: Union[float, int]) -> None:
    """Log a metric to MLflow.

    Parameters
    ----------
    key : str
        Metric name.
    value : Union[float, int]
        Metric value.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Ensure we have an active run
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run. Starting a new run.")
            start_run()

        # Check if value is valid for MLflow
        if value is not None and not pd.isna(value):
            mlflow.log_metric(key, float(value))
            logger.debug(f"Logged metric: {key}={value}")
        else:
            logger.debug(f"Skipped logging invalid metric value: {key}={value}")
    except Exception as e:
        logger.error(f"Error logging metric: {e}")


def log_model(model: Any, model_name: str) -> None:
    """Log a model to MLflow.

    Parameters
    ----------
    model : Any
        Model to log.
    model_name : str
        Name of the model.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Ensure we have an active run
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run. Starting a new run.")
            start_run()

        mlflow.sklearn.log_model(model, model_name)
        logger.debug(f"Logged model: {model_name}")
    except Exception as e:
        logger.error(f"Error logging model: {e}")


def log_artifacts(local_dir: str) -> None:
    """Log artifacts to MLflow.

    Parameters
    ----------
    local_dir : str
        Local directory containing artifacts to log.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Ensure we have an active run
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run. Starting a new run.")
            start_run()

        mlflow.log_artifacts(local_dir)
        logger.debug(f"Logged artifacts from: {local_dir}")
    except Exception as e:
        logger.error(f"Error logging artifacts: {e}")


def log_model_performance(
    model_name: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Log model performance metrics to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the model.
    metrics : Dict[str, float]
        Performance metrics.
    params : Dict[str, Any], optional (default=None)
        Model parameters.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Ensure we have an active run
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run. Starting a new run.")
            start_run()

        # Log metrics
        for key, value in metrics.items():
            if value is not None and not pd.isna(value):  # Only log valid values
                try:
                    mlflow.log_metric(f"{model_name}_{key}", float(value))
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert metric {key}={value} to float, skipping.")

        # Log parameters
        if params:
            prefixed_params = {}
            for k, v in params.items():
                if v is None:
                    prefixed_params[f"{model_name}_{k}"] = "None"
                elif isinstance(v, (str, int, float, bool)):
                    prefixed_params[f"{model_name}_{k}"] = v
                else:
                    prefixed_params[f"{model_name}_{k}"] = str(v)

            mlflow.log_params(prefixed_params)

        logger.debug(f"Logged performance for model: {model_name}")
    except Exception as e:
        logger.error(f"Error logging model performance: {e}")


def log_dataframe(df: pd.DataFrame, artifact_name: str) -> None:
    """Log a DataFrame as an artifact to MLflow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to log.
    artifact_name : str
        Name of the artifact.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
        return

    try:
        # Ensure we have an active run
        active_run = mlflow.active_run()
        if not active_run:
            logger.warning("No active MLflow run. Starting a new run.")
            active_run = start_run()

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Ensure the directory exists
            os.makedirs(tmp_dir, exist_ok=True)

            # Save DataFrame to CSV
            file_path = os.path.join(tmp_dir, f"{artifact_name}.csv")
            df.to_csv(file_path, index=False)

            # Log the file as an artifact
            mlflow.log_artifact(file_path)
            logger.debug(f"Logged DataFrame as artifact: {artifact_name}")
    except Exception as e:
        logger.error(f"Error logging DataFrame: {e}")
