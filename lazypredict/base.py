"""
Base module for lazypredict.

This module provides the base class for all lazy model implementations.
"""

import logging

import numpy as np
import pandas as pd

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class Lazy:
    """
    Base class for all lazy model implementations.

    Parameters
    ----------
    verbose : int, default=1
        Verbosity level (0: no output, 1: minimal output, 2: detailed output)

    ignore_warnings : bool, default=True
        Whether to ignore warnings during model training

    random_state : int, default=42
        Random state for reproducibility
    """

    def __init__(self, verbose=1, ignore_warnings=True, random_state=42):
        if not isinstance(verbose, int) or verbose < 0:
            raise ValueError("verbose must be a non-negative integer")

        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.random_state = random_state

    def _check_gpu_availability(self):
        """Check if GPU is available for computation."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            logger.warning(
                "torch not installed. GPU acceleration not available."
            )
            return False

    def _check_data(self, X_train, X_test, y_train, y_test):
        """
        Check and prepare data for model training.

        Converts pandas DataFrames to numpy arrays if needed.

        Parameters
        ----------
        X_train : array-like
            Training features

        X_test : array-like
            Test features

        y_train : array-like
            Training target

        y_test : array-like
            Test target

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple
            Prepared data arrays
        """
        # Convert pandas DataFrames to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()

        # Convert pandas Series to numpy arrays
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy()
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_test = y_test.to_numpy()

        # Reshape y if needed
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.ravel()

        return X_train, X_test, y_train, y_test

    def fit(self, X_train, X_test, y_train, y_test):
        """
        Fit models.

        This is an abstract method that should be implemented by subclasses.

        Parameters
        ----------
        X_train : array-like
            Training features

        X_test : array-like
            Test features

        y_train : array-like
            Training target

        y_test : array-like
            Test target

        Returns
        -------
        scores : pandas.DataFrame
            DataFrame with performance metrics for each model

        predictions : dict
            Dictionary with predictions for each model
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses"
        )

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        Train and return all models.

        This is an abstract method that should be implemented by subclasses.

        Parameters
        ----------
        X_train : array-like
            Training features

        X_test : array-like
            Test features

        y_train : array-like
            Training target

        y_test : array-like
            Test target

        Returns
        -------
        models : dict
            Dictionary with trained models
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses"
        )
