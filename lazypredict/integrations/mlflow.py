"""MLflow integration helpers for LazyPredict."""

import inspect
import logging
import os
from typing import Any, Optional

from lazypredict._optional import warn_if_broken

logger = logging.getLogger("lazypredict")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception as exc:  # missing, or installed but broken
    MLFLOW_AVAILABLE = False
    warn_if_broken("mlflow", exc)


def is_mlflow_tracking_enabled() -> bool:
    """Check if MLflow tracking is enabled via the MLFLOW_TRACKING_URI environment variable."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return MLFLOW_AVAILABLE and tracking_uri is not None


def setup_mlflow() -> bool:
    """Point MLflow at ``MLFLOW_TRACKING_URI`` when that variable is set.

    Autologging is deliberately not enabled: it is a process-wide setting
    that would change how the user's own code is logged.

    Returns
    -------
    bool
        True if MLflow was successfully configured, False otherwise.
    """
    if is_mlflow_tracking_enabled():
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        return True
    return False


def start_run(mlflow_module: Any, run_name: str) -> Any:
    """Start a run for one model, nested under the caller's run if one is active."""
    return mlflow_module.start_run(run_name=run_name, nested=True)


def log_sklearn_model(
    mlflow_module: Any, model: Any, artifact_name: str, signature: Optional[Any] = None
) -> None:
    """Log *model* as an artifact of the active run without registering it.

    MLflow 3 names the artifact with ``name=``; MLflow 2 only accepts
    ``artifact_path=``.
    """
    log_model = mlflow_module.sklearn.log_model
    key = "name" if "name" in inspect.signature(log_model).parameters else "artifact_path"
    log_model(model, **{key: artifact_name}, signature=signature)
