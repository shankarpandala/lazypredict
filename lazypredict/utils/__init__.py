"""Utilities for lazy model selection."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    BaseLazy,
    check_X_y,
    get_model_name,
)
from .gpu import (
    get_best_model,
    get_cpu_model,
    is_cuml_available,
    is_gpu_available,
)
from .metrics import (
    get_classification_metrics,
    get_regression_metrics,
)
from .mlflow_utils import (
    configure_mlflow,
    end_run,
    log_artifacts,
    log_dataframe,
    log_metric,
    log_model,
    log_model_performance,
    log_params,
    start_run,
)
from .preprocessing import (
    categorical_cardinality_threshold,
    create_preprocessor,
    get_card_split,
)

logger = logging.getLogger("lazypredict.utils")


def get_model_name(model_class: Any) -> str:
    """Get the name of a model class.

    Parameters
    ----------
    model_class : Any
        The model class or string name.

    Returns
    -------
    str
        Name of the model class.
    """
    try:
        if isinstance(model_class, str):
            # Return string directly if it's already a string
            return model_class
        elif hasattr(model_class, "__name__"):
            return str(model_class.__name__)
        elif hasattr(model_class, "__class__"):
            return str(model_class.__class__.__name__)
        else:
            return str(model_class)
    except Exception as e:
        logger.warning(f"Error getting model name: {e}")
        return str(model_class)


def check_X_y(X, y):
    """
    Check and prepare X and y for model training.

    Converts pandas DataFrames to numpy arrays if needed.

    Parameters
    ----------
    X : array-like
        Features

    y : array-like
        Target

    Returns
    -------
    X, y : tuple
        Prepared data arrays
    """
    # Convert pandas DataFrames to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Convert pandas Series to numpy arrays
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()

    # Reshape y if needed
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.ravel()

    return X, y


__all__ = [
    # Base utils
    "BaseLazy",
    "check_X_y",
    "get_model_name",
    # GPU utils
    "is_gpu_available",
    "is_cuml_available",
    "get_cpu_model",
    "get_best_model",
    # Metrics
    "get_regression_metrics",
    "get_classification_metrics",
    # MLflow utils
    "configure_mlflow",
    "end_run",
    "log_artifacts",
    "log_dataframe",
    "log_metric",
    "log_model",
    "log_model_performance",
    "log_params",
    "start_run",
    # Preprocessing
    "create_preprocessor",
    "categorical_cardinality_threshold",
    "get_card_split",
]

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
