import pandas as pd
try:
    import cudf
except ImportError:
    cudf = None
try:
    import polars as pl
except ImportError:
    pl = None

class BackendManager:
    """
    BackendManager for switching between different dataframe libraries.

    This class allows users to choose between Pandas, CuDF, and Polars, depending on hardware
    availability and performance preferences.

    Attributes
    ----------
    backend : str
        The backend to use ('pandas', 'cudf', or 'polars').

    Methods
    -------
    set_backend(backend):
        Sets the dataframe backend.
    get_dataframe(data):
        Returns a dataframe in the specified backend format.
    """

    def __init__(self, backend="pandas"):
        """
        Parameters
        ----------
        backend : str, optional
            The backend to use ('pandas', 'cudf', or 'polars'). Default is 'pandas'.
        """
        self.backend = backend.lower()

    def set_backend(self, backend):
        """Set the dataframe backend."""
        self.backend = backend.lower()

    def get_dataframe(self, data):
        """
        Convert data to the specified backend's dataframe format.

        Parameters
        ----------
        data : array-like
            Data to convert to the specified backend format.

        Returns
        -------
        DataFrame
            Data in the selected backend's dataframe format.
        """
        if self.backend == "pandas":
            return pd.DataFrame(data)
        elif self.backend == "cudf" and cudf is not None:
            return cudf.DataFrame.from_pandas(pd.DataFrame(data))
        elif self.backend == "polars" and pl is not None:
            return pl.DataFrame(data)
        else:
            raise ValueError(f"Backend '{self.backend}' not supported or unavailable.")
