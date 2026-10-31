"""Preprocessing utilities — encoders, transformers, and cardinality splitting."""

import logging
import numbers
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from lazypredict.config import VALID_ENCODERS
from lazypredict._optional import warn_if_broken

logger = logging.getLogger("lazypredict")

# Optional category_encoders import with fallback
try:
    from category_encoders import BinaryEncoder, TargetEncoder
    CATEGORY_ENCODERS_AVAILABLE = True
except Exception as exc:  # missing, or installed but broken
    CATEGORY_ENCODERS_AVAILABLE = False
    warn_if_broken("category_encoders", exc)

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


def _is_categorical_dtype(dtype) -> bool:
    """True for ``object``, ``string``/``str`` and ``category`` columns."""
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def _category_label(value: Any) -> str:
    """String label of a category value; equal numbers get equal labels.

    A column of integer codes becomes float as soon as one value is missing,
    so ``1`` and ``1.0`` must map to the same label for the categories learnt
    on the training data to be recognised in the test data.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value)


def _column_as_strings(column: pd.Series) -> pd.Series:
    """Return *column* as ``object`` values of ``str``, with ``np.nan`` for missing."""
    missing = column.isna()
    if isinstance(column.dtype, pd.CategoricalDtype):
        labels = column.cat.categories
        values = column.astype(object)
        if pd.api.types.infer_dtype(labels, skipna=True) != "string":
            values = values.map(_category_label, na_action="ignore")
    elif pd.api.types.infer_dtype(column, skipna=True) in ("string", "empty"):
        values = column.astype(object)
    else:
        values = column.astype(object).map(_category_label, na_action="ignore")
    return values.where(~missing, np.nan)


def categories_as_strings(X) -> np.ndarray:
    """Normalise categorical input before imputation and encoding.

    ``category``, nullable ``string``, pandas 3 ``str`` and mixed-type
    ``object`` columns become one ``object`` array of ``str`` values with
    ``np.nan`` for every kind of missing value (``None``, ``pd.NA``, ``NaN``),
    so they are imputed and encoded the same way even when the training and
    test data use different dtypes.
    """
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    columns = [
        _column_as_strings(frame.iloc[:, i]).to_numpy(dtype=object)
        for i in range(frame.shape[1])
    ]
    if not columns:
        return np.empty((len(frame), 0), dtype=object)
    return np.column_stack(columns)


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

    return Pipeline(
        steps=[
            (
                "as_strings",
                FunctionTransformer(categories_as_strings, feature_names_out="one-to-one"),
            ),
            ("imputer", imputer),
            ("encoding", encoder),
        ]
    )


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
    """Return ``X_train`` and ``X_test`` as DataFrames with the same column names.

    Array-likes (NumPy arrays, lists) take the training DataFrame's column
    names, or ``feature_0 .. feature_n`` when the training data has none; in
    that case a DataFrame ``X_test`` is relabelled by position. ``X_test``
    keeps its index whenever it is a DataFrame. Numeric columns of ``object``
    arrays (e.g. ``df.to_numpy()`` with a text column) get numeric dtypes, and
    non-string column labels become strings.
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = with_string_column_names(X_train)
        if isinstance(X_test, pd.DataFrame):
            X_test = with_string_column_names(X_test)
        else:
            X_test = _array_to_frame(X_test, X_train.columns)
        return X_train, X_test

    X_train = _array_to_frame(X_train)
    columns = pd.Index([f"feature_{i}" for i in range(X_train.shape[1])])
    X_train.columns = columns
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.set_axis(columns, axis=1)
    else:
        X_test = _array_to_frame(X_test, columns)
    return X_train, X_test


def _array_to_frame(X: Any, columns: Any = None) -> pd.DataFrame:
    """DataFrame from an array-like, with numeric dtypes for numeric columns."""
    return pd.DataFrame(np.asarray(X), columns=columns).infer_objects()


def with_string_column_names(X: pd.DataFrame) -> pd.DataFrame:
    """Return *X* with ``str`` column labels.

    scikit-learn's ``ColumnTransformer`` treats integer labels as positions,
    so the columns ``[1, 2, 3]`` of a headerless CSV would select the wrong
    columns or fail.
    """
    if all(isinstance(column, str) for column in X.columns):
        return X
    return X.rename(columns=str)


def _is_temporal_dtype(dtype: Any) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype))


def times_as_numbers(X) -> np.ndarray:
    """Encode datetime columns as days since 1970 and timedeltas as seconds.

    Missing values (``NaT``) become ``np.nan`` for the imputer that follows.
    """
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    columns = []
    for i in range(frame.shape[1]):
        column = frame.iloc[:, i]
        if pd.api.types.is_timedelta64_dtype(column.dtype):
            values = pd.to_timedelta(column).dt.total_seconds()
        else:
            moments = pd.to_datetime(column, utc=True)
            values = (moments - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(days=1)
        columns.append(np.asarray(values, dtype=float))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0))


def build_preprocessor(
    X_train: pd.DataFrame, categorical_encoder: str
) -> ColumnTransformer:
    """Build a ColumnTransformer for the given data.

    Numeric and boolean columns are imputed and scaled, categorical columns
    are imputed and encoded, and datetime/timedelta columns are converted to
    numbers (days since 1970, seconds) and then treated as numeric.
    """
    temporal_features = pd.Index(
        [col for col, dtype in X_train.dtypes.items() if _is_temporal_dtype(dtype)]
    )
    # timedelta64 counts as np.number in pandas; it goes to the temporal branch.
    numeric_features = X_train.select_dtypes(include=[np.number, "bool"]).columns.difference(
        temporal_features, sort=False
    )
    categorical_features = pd.Index(
        [col for col, dtype in X_train.dtypes.items() if _is_categorical_dtype(dtype)]
    )

    categorical_low, categorical_high = get_card_split(X_train, categorical_features)

    cat_low = get_categorical_encoder(categorical_encoder, cardinality="low")
    cat_high = get_categorical_encoder(categorical_encoder, cardinality="high")

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            (
                "temporal",
                Pipeline(
                    steps=[
                        ("as_numbers", FunctionTransformer(times_as_numbers, feature_names_out="one-to-one")),
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                temporal_features,
            ),
            ("categorical_low", cat_low, categorical_low),
            ("categorical_high", cat_high, categorical_high),
        ]
    )
