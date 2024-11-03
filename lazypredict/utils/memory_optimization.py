# lazypredict/utils/memory_optimization.py

import pandas as pd

def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes the memory usage of a DataFrame by downcasting numeric types.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with optimized memory usage.
    """
    for col in df.select_dtypes(include=['int', 'float']).columns:
        col_type = df[col].dtypes
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
