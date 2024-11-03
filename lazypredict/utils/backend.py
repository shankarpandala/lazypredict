# lazypredict/utils/backend.py

import os
import logging

# No direct imports of pandas, cudf, or polars at the top
logger = logging.getLogger(__name__)

# Backend choice: 'pandas', 'cudf', 'polars'
BACKEND_ENV_VARIABLE = 'LAZYPREDICT_BACKEND'

class Backend:
    _backend = None

    @staticmethod
    def initialize_backend(use_gpu: bool = False):
        """
        Initializes the backend based on user preference.
        If use_gpu is True, it attempts to use cuDF if available.
        """
        backend_choice = os.getenv(BACKEND_ENV_VARIABLE, 'pandas').lower()

        if use_gpu:
            try:
                import cudf
                Backend._backend = cudf
                logger.info("Using cuDF as the backend.")
                return
            except ImportError:
                logger.warning("cuDF not available, falling back to the selected backend.")

        if backend_choice == 'cudf':
            try:
                import cudf
                Backend._backend = cudf
                logger.info("Using cuDF as the backend.")
            except ImportError:
                logger.error("cuDF not available. Install RAPIDS or set the backend to another option.")
                raise ImportError("cuDF is not available. Install RAPIDS or choose another backend.")

        elif backend_choice == 'polars':
            try:
                import polars as pl
                Backend._backend = pl
                logger.info("Using polars as the backend.")
            except ImportError:
                logger.error("Polars is not available. Install polars or set the backend to another option.")
                raise ImportError("Polars is not available. Install polars or choose another backend.")

        else:
            import pandas as pd
            Backend._backend = pd
            logger.info("Using pandas as the backend.")

    @staticmethod
    def get_backend():
        """
        Returns the backend object.
        """
        if Backend._backend is None:
            Backend.initialize_backend()
        return Backend._backend

    @staticmethod
    def DataFrame(*args, **kwargs):
        """
        Returns a DataFrame object from the selected backend.
        """
        backend = Backend.get_backend()
        return backend.DataFrame(*args, **kwargs)

    @staticmethod
    def Series(*args, **kwargs):
        """
        Returns a Series object from the selected backend.
        """
        backend = Backend.get_backend()
        return backend.Series(*args, **kwargs)
