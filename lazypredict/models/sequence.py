"""
Sequence prediction module for LazyPredict.
"""

from ..base import Lazy


class LazySequencePredictor(Lazy):
    """
    Automated machine learning for sequence prediction.

    This class automates the process of training and evaluating multiple sequence prediction models.

    Parameters
    ----------
    verbose : int, default=1
        Verbosity level (0: no output, 1: minimal output, 2: detailed output)

    ignore_warnings : bool, default=True
        Whether to ignore warnings during model training

    custom_metric : callable, default=None
        Custom metric function (func(y_true, y_pred) should return a score)

    models : list, default=None
        List of model classes to use, if None all available models are used

    random_state : int, default=42
        Random state for reproducibility
    """

    def __init__(
        self,
        verbose=1,
        ignore_warnings=True,
        custom_metric=None,
        models=None,
        random_state=42,
    ):
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            random_state=random_state,
        )
        self.custom_metric = custom_metric
        self.models_list = models
        self.models = {}

    def fit(self, X, y):
        """
        Fit all sequence prediction models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps, n_features)
            Sequential features

        y : array-like
            Target

        Returns
        -------
        scores : pandas.DataFrame
            DataFrame with performance metrics for each model
        """
        raise NotImplementedError("LazySequencePredictor.fit() is not implemented yet")

    def provide_models(self, X, y):
        """
        Train and return all models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timesteps, n_features)
            Sequential features

        y : array-like
            Target

        Returns
        -------
        models : dict
            Dictionary with trained models
        """
        raise NotImplementedError("LazySequencePredictor.provide_models() is not implemented yet")
