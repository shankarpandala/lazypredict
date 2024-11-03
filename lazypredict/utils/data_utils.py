# lazypredict/utils/data_utils.py

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

# Import Backend only when needed to avoid circular import issues
from .backend import Backend

logger = logging.getLogger(__name__)

def prepare_data(X: Any) -> Any:
    """
    Prepares input data by converting it to the appropriate DataFrame type based on the backend.

    Args:
        X (Any): Input data.

    Returns:
        Any: Prepared DataFrame.
    """
    DataFrame = Backend.DataFrame

    if not isinstance(X, DataFrame):
        try:
            X = DataFrame(X)
        except Exception as e:
            logger.error(f"Failed to convert data to DataFrame: {e}")
            raise ValueError("Input data cannot be converted to a DataFrame.")
    return X


def get_card_split(df: Any, categorical_features: List[str], threshold: int = 10) -> Tuple[List[str], List[str]]:
    """
    Splits categorical features into low and high cardinality based on a threshold.

    Args:
        df (Any): Input DataFrame.
        categorical_features (List[str]): List of categorical feature names.
        threshold (int, optional): Cardinality threshold. Defaults to 10.

    Returns:
        Tuple[List[str], List[str]]: Lists of low and high cardinality features.
    """
    low_cardinality = []
    high_cardinality = []

    for col in categorical_features:
        num_unique = df[col].nunique()
        if num_unique <= threshold:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)
    return low_cardinality, high_cardinality
