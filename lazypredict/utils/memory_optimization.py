import pandas as pd

class MemoryOptimizer:
    """
    MemoryOptimizer for optimizing dataframe memory usage.

    Methods
    -------
    optimize_memory(df):
        Optimize memory usage by downcasting numeric columns.
    """

    @staticmethod
    def optimize_memory(df):
        """
        Optimize memory usage of a dataframe by downcasting numeric columns.

        Parameters
        ----------
        df : DataFrame
            Dataframe to optimize.

        Returns
        -------
        DataFrame
            Optimized dataframe with reduced memory usage.
        """
        for col in df.select_dtypes(include=['float', 'int']).columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="float")
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="integer")
        return df
