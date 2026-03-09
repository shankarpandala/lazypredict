"""Preprocessing utilities — encoders, transformers, and cardinality splitting."""

import logging
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lazypredict.config import VALID_ENCODERS

logger = logging.getLogger("lazypredict")

# Optional category_encoders import with fallback
try:
    from category_encoders import BinaryEncoder, TargetEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except ImportError:
    CATEGORY_ENCODERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default preprocessing pipelines
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder()),
    ]
)


def get_categorical_encoder(
    encoder_type: str = "onehot", cardinality: str = "low"
) -> Pipeline:
    """Get categorical encoder pipeline based on encoder type and cardinality.

    Parameters
    ----------
    encoder_type : str, optional (default='onehot')
        Type of encoder: 'onehot', 'ordinal', 'target', or 'binary'.
    cardinality : str, optional (default='low')
        Cardinality level: 'low' or 'high'.

    Returns
    -------
    Pipeline
        Sklearn pipeline with imputer and encoder.

    Raises
    ------
    ValueError
        If *encoder_type* is not one of the recognised values.
    """
    imputer = SimpleImputer(strategy="constant", fill_value="missing")

    if encoder_type == "onehot":
        if cardinality == "low":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
    elif encoder_type == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    elif encoder_type == "target":
        if not CATEGORY_ENCODERS_AVAILABLE:
            logger.warning(
                "category_encoders not installed. Falling back to ordinal encoding. "
                "Install with: pip install category_encoders"
            )
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        else:
            encoder = TargetEncoder(handle_unknown="value", handle_missing="value")
    elif encoder_type == "binary":
        if not CATEGORY_ENCODERS_AVAILABLE:
            logger.warning(
                "category_encoders not installed. Falling back to onehot encoding. "
                "Install with: pip install category_encoders"
            )
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = BinaryEncoder(handle_unknown="value", handle_missing="value")
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Choose from {VALID_ENCODERS!r}"
        )

    return Pipeline(steps=[("imputer", imputer), ("encoding", encoder)])


def get_card_split(
    df: pd.DataFrame, cols: Union[pd.Index, list], n: int = 11
) -> Tuple[pd.Index, pd.Index]:
    """Split categorical columns into two lists based on cardinality.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : array-like
        Categorical columns to evaluate.
    n : int, optional (default=11)
        Columns with more than *n* unique values are considered high cardinality.

    Returns
    -------
    card_low : pandas.Index
        Columns with cardinality <= *n*.
    card_high : pandas.Index
        Columns with cardinality > *n*.
    """
    cols = pd.Index(cols)
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def prepare_dataframes(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert ndarrays to DataFrames with consistent column names."""
    if isinstance(X_train, np.ndarray):
        cols = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)
    return X_train, X_test


def build_preprocessor(
    X_train: pd.DataFrame, categorical_encoder: str
) -> ColumnTransformer:
    """Build a ColumnTransformer for the given data."""
    numeric_features = X_train.select_dtypes(include=[np.number, "bool"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    categorical_low, categorical_high = get_card_split(X_train, categorical_features)

    cat_low = get_categorical_encoder(categorical_encoder, cardinality="low")
    cat_high = get_categorical_encoder(categorical_encoder, cardinality="high")

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical_low", cat_low, categorical_low),
            ("categorical_high", cat_high, categorical_high),
        ]
    )
