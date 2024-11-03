from sklearn.inspection import permutation_importance

class PermutationImportance:
    """
    PermutationImportance for estimating feature importance by shuffling feature values.

    Attributes
    ----------
    model : object
        The trained model for which permutation importance is computed.
    X : DataFrame
        Data on which to compute permutation importance.
    y : array-like
        True labels for the data.

    Methods
    -------
    compute_importance():
        Computes and returns permutation importances.
    """

    def __init__(self, model, X, y):
        """
        Parameters
        ----------
        model : object
            Trained model compatible with permutation importance.
        X : DataFrame
            Input data.
        y : array-like
            True labels for the data.
        """
        self.model = model
        self.X = X
        self.y = y

    def compute_importance(self, n_repeats=10):
        """
        Compute permutation importances for the model.

        Parameters
        ----------
        n_repeats : int, optional
            Number of repetitions for shuffling each feature. Default is 10.

        Returns
        -------
        DataFrame
            Permutation importance scores for each feature.
        """
        results = permutation_importance(self.model, self.X, self.y, n_repeats=n_repeats)
        return pd.DataFrame({
            "feature": self.X.columns,
            "importance_mean": results.importances_mean,
            "importance_std": results.importances_std
        })
