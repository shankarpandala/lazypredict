from sklearn.inspection import plot_partial_dependence
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
        plot_partial_dependence(self.model, self.X, features, grid_resolution=50)
        plt.show()
