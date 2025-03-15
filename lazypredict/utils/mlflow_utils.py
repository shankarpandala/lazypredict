"""
MLflow utilities for lazypredict.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("lazypredict.mlflow")

# Global variables
GLOBAL_MLFLOW_TRACKING_URI = None
ACTIVE_RUN = None

def configure_mlflow(tracking_uri: Optional[str] = None) -> None:
    """Configure MLflow tracking.
    
    Parameters
    ----------
    tracking_uri : str, optional (default=None)
        MLflow tracking URI. If None, will use the global tracking URI.
    """
    global GLOBAL_MLFLOW_TRACKING_URI
    
    try:
        import mlflow
        
        # Set the tracking URI
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
            GLOBAL_MLFLOW_TRACKING_URI = tracking_uri
        elif GLOBAL_MLFLOW_TRACKING_URI is not None:
            mlflow.set_tracking_uri(GLOBAL_MLFLOW_TRACKING_URI)
        elif "MLFLOW_TRACKING_URI" in os.environ:
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            GLOBAL_MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
        
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")

def start_run(
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = "lazypredict",
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Start a new MLflow run.
    
    Parameters
    ----------
    run_name : str, optional (default=None)
        Name of the run.
    experiment_name : str, optional (default="lazypredict")
        Name of the experiment.
    tags : Dict[str, str], optional (default=None)
        Tags to set on the run.
    """
    global ACTIVE_RUN
    
    try:
        import mlflow
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        ACTIVE_RUN = mlflow.start_run(run_name=run_name)
        
        # Set tags
        if tags:
            mlflow.set_tags(tags)
            
        logger.info(f"Started MLflow run: {ACTIVE_RUN.info.run_id}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
    except Exception as e:
        logger.error(f"Error starting MLflow run: {e}")

def end_run() -> None:
    """End the active MLflow run."""
    global ACTIVE_RUN
    
    try:
        import mlflow
        
        mlflow.end_run()
        if ACTIVE_RUN is not None:
            logger.info(f"Ended MLflow run: {ACTIVE_RUN.info.run_id}")
            ACTIVE_RUN = None
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
    except Exception as e:
        logger.error(f"Error ending MLflow run: {e}")

def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Parameters to log.
    """
    try:
        import mlflow
        
        mlflow.log_params(params)
        logger.debug(f"Logged parameters: {params}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
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
    try:
        import mlflow
        
        mlflow.log_metric(key, value)
        logger.debug(f"Logged metric: {key}={value}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
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
    try:
        import mlflow.sklearn
        
        if mlflow.active_run() is not None:
            mlflow.sklearn.log_model(model, model_name)
            logger.debug(f"Logged model: {model_name}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
    except Exception as e:
        logger.error(f"Error logging model: {e}")

def log_artifacts(local_dir: str) -> None:
    """Log artifacts to MLflow.
    
    Parameters
    ----------
    local_dir : str
        Local directory containing artifacts to log.
    """
    try:
        import mlflow
        
        if mlflow.active_run() is not None:
            mlflow.log_artifacts(local_dir)
            logger.debug(f"Logged artifacts from: {local_dir}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
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
    try:
        import mlflow
        
        if mlflow.active_run() is not None:
            # Create a new run for this model
            with mlflow.start_run(run_name=model_name, nested=True):
                # Log model name
                mlflow.set_tag("model", model_name)
                
                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                # Log parameters
                if params:
                    mlflow.log_params(params)
                    
                logger.debug(f"Logged performance for model: {model_name}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
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
    try:
        import mlflow
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, f"{artifact_name}.csv")
            df.to_csv(file_path, index=False)
            mlflow.log_artifact(file_path, artifact_name)
            logger.debug(f"Logged DataFrame as artifact: {artifact_name}")
    except ImportError:
        logger.warning("MLflow not installed. Experiment tracking disabled.")
    except Exception as e:
        logger.error(f"Error logging DataFrame: {e}") 