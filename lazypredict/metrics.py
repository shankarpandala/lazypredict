"""Metric helper functions for LazyPredict."""


def adjusted_rsquared(r2: float, n: int, p: int) -> float:
    """Calculate adjusted R-squared.

    Parameters
    ----------
    r2 : float
        R-squared value.
    n : int
        Number of observations.
    p : int
        Number of predictors.

    Returns
    -------
    float
        Adjusted R-squared value.
    """
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))
