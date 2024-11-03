from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

class PartialDependencePlot:
    """
    PartialDependencePlot for visualizing the marginal effect of features on predictions.

    Attributes
    ----------
    model : object
        The trained model for which partial dependence is computed.
    X : DataFrame
        Data on which to compute partial dependence.

    Methods
    -------
    plot(features):
        Plot partial dependence for specified features.
    """

    def __init__(self, model, X):
        """
        Parameters
        ----------
        model : object
            Trained model compatible with partial dependence plot.
        X : DataFrame
            Input data.
        """
        self.model = model
        self.X = X

    def plot(self, features):
        """
        Plot partial dependence for specified features.

        Parameters
        ----------
        features : list of str
            List of feature names to plot.

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(self.model, self.X, features, ax=ax, grid_resolution=50)
        plt.show()
