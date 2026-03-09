"""MLflow integration helpers for LazyPredict."""

import logging
import os

logger = logging.getLogger("lazypredict")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def is_mlflow_tracking_enabled() -> bool:
    """Check if MLflow tracking is enabled via the MLFLOW_TRACKING_URI environment variable."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return MLFLOW_AVAILABLE and tracking_uri is not None


def setup_mlflow() -> bool:
    """Initialize MLflow if tracking URI is set through environment variable.

    Returns
    -------
    bool
        True if MLflow was successfully configured, False otherwise.
    """
    if is_mlflow_tracking_enabled():
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.autolog()
        return True
    return False
