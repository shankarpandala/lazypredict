"""Distributed computing utilities for LazyPredict.

Provides Dask joblib backend registration and PySpark/Dask DataFrame
auto-conversion.
"""

import logging
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("lazypredict")


# ---------------------------------------------------------------------------
# PySpark / Dask DataFrame auto-conversion
# ---------------------------------------------------------------------------


def auto_convert_dataframe(
    data: Any, name: str = "data"
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Auto-convert PySpark or Dask DataFrames to pandas.

    Parameters
    ----------
    data : any
        Input data — may be pandas, PySpark, Dask, or numpy.
    name : str
        Label for log messages.

    Returns
    -------
    pandas.DataFrame, pandas.Series, or numpy.ndarray
        Converted data.
    """
    cls_name = type(data).__module__ + "." + type(data).__name__

    # PySpark DataFrame
    if "pyspark.sql" in cls_name and hasattr(data, "toPandas"):
        logger.warning(
            "Auto-converting PySpark DataFrame (%s) to pandas via .toPandas(). "
            "This collects all data to the driver node.",
            name,
        )
        return data.toPandas()

    # Dask DataFrame / Series
    if "dask.dataframe" in cls_name and hasattr(data, "compute"):
        logger.warning(
            "Auto-converting Dask DataFrame (%s) to pandas via .compute(). "
            "This materializes the full dataset in memory.",
            name,
        )
        return data.compute()

    # Dask Array
    if "dask.array" in cls_name and hasattr(data, "compute"):
        logger.warning(
            "Auto-converting Dask Array (%s) to numpy via .compute().",
            name,
        )
        return data.compute()

    return data


# ---------------------------------------------------------------------------
# Dask joblib backend for cross-validation
# ---------------------------------------------------------------------------


_dask_backend_registered = False


def register_dask_backend() -> bool:
    """Register Dask as a joblib parallel backend if available.

    Call this before cross-validation to distribute folds across
    Dask workers. Returns True if registration succeeded.

    Usage
    -----
    >>> from lazypredict.distributed import register_dask_backend
    >>> if register_dask_backend():
    ...     # Now cross_validate will use Dask workers
    ...     pass
    """
    global _dask_backend_registered
    if _dask_backend_registered:
        return True

    try:
        from dask.distributed import Client
        import joblib

        # Register dask as a parallel backend
        try:
            from joblib import parallel_config
            # joblib >= 1.3 uses parallel_config
            _dask_backend_registered = True
        except ImportError:
            pass

        # Try the older registration approach
        try:
            from dask.distributed import Client
            # Check if a client is already running
            try:
                client = Client.current()
                logger.info(
                    "Dask client found at %s — using Dask for parallel CV.",
                    client.dashboard_link or "local",
                )
            except ValueError:
                # No client running, create a local one
                client = Client(processes=False, silence_logs=logging.WARNING)
                logger.info(
                    "Created local Dask client for parallel CV."
                )
            _dask_backend_registered = True
            return True
        except ImportError:
            return False

    except ImportError:
        logger.info(
            "dask.distributed not available — using default joblib backend."
        )
        return False


def get_dask_joblib_context():
    """Return a joblib parallel_config context manager for Dask, or None.

    Usage
    -----
    >>> ctx = get_dask_joblib_context()
    >>> if ctx:
    ...     with ctx:
    ...         cross_validate(...)
    """
    if not _dask_backend_registered:
        if not register_dask_backend():
            return None

    try:
        from joblib import parallel_config
        return parallel_config(backend="dask")
    except (ImportError, Exception):
        return None
