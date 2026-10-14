"""Preprocessing utilities — encoders, transformers, and cardinality splitting."""

import inspect
import logging
import numbers
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import _check_feature_names_in

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


def _datetime_label(value):
    """Preserve valid date labels outside pandas' nanosecond bounds."""
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC")
    except (pd.errors.OutOfBoundsDatetime, OverflowError):
        timestamp = value
        if isinstance(value, np.datetime64):
            try:
                narrowed = value.astype("datetime64[us]")
            except OverflowError:
                return f"datetime:{np.datetime_as_string(value, unit='auto')}"
            if narrowed.astype(value.dtype) != value:
                return f"datetime:{np.datetime_as_string(value, unit='auto')}"
            timestamp = narrowed.item()
        if not isinstance(timestamp, datetime):
            return f"datetime:{np.datetime_as_string(value, unit='auto')}"
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(timezone.utc)
    return f"datetime:{timestamp.isoformat()}"


def _timedelta_label(value):
    """Serialize fixed durations exactly without bounded-width casts."""
    if isinstance(value, pd.Timedelta):
        value = value.asm8
    if isinstance(value, np.timedelta64):
        unit, multiplier = np.datetime_data(value.dtype)
        scales = {"W": 604800 * 10**18, "D": 86400 * 10**18, "h": 3600 * 10**18,
                  "m": 60 * 10**18, "s": 10**18, "ms": 10**15, "us": 10**12,
                  "ns": 10**9, "ps": 10**6, "fs": 10**3, "as": 1}
        if unit not in scales:
            return f"{type(value).__name__}:{value}"
        total = int(value.view("i8")) * multiplier * scales[unit]
    else:
        total = ((value.days * 86400 + value.seconds) * 1000000 + value.microseconds) * 10**12
    days, remainder = divmod(total, 86400 * 10**18)
    hours, remainder = divmod(remainder, 3600 * 10**18)
    minutes, remainder = divmod(remainder, 60 * 10**18)
    seconds, attoseconds = divmod(remainder, 10**18)
    fraction = f".{attoseconds:018d}".rstrip("0") if attoseconds else ""
    return f"timedelta:P{days}DT{hours}H{minutes}M{seconds}{fraction}S"


def _rational_label(value):
    """Match builtin labels only when conversion preserves the exact ratio."""
    numerator, denominator = (
        value.as_integer_ratio() if isinstance(value, Decimal) else (value.numerator, value.denominator)
    )
    if denominator == 1:
        return f"number:{numerator}"
    try:
        approximate = float(value)
    except OverflowError:
        approximate = np.inf
    if np.isfinite(approximate) and approximate.as_integer_ratio() == (numerator, denominator):
        return f"number:{approximate!r}"
    return f"number:{numerator}/{denominator}"


def _numeric_label(value):
    """Preserve exact numeric equality without rounding distinct categories."""
    if isinstance(value, numbers.Integral):
        return f"number:{int(value)}"
    if isinstance(value, (float, np.floating)):
        label = str(int(value)) if np.isfinite(value) and value % 1 == 0 else repr(float(value))
        return f"number:{label}"
    if isinstance(value, Decimal) and not value.is_finite():
        return f"number:{float(value)!r}"
    if isinstance(value, (Decimal, numbers.Rational)):
        return _rational_label(value)
    if isinstance(value, (complex, np.complexfloating)) and value.imag == 0:
        return _numeric_label(value.real)
    return f"{type(value).__name__}:{value}"


def _categorical_label(value):
    """Use consistent typed labels across separately inferred categories."""
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime)):
        return _datetime_label(value)
    if isinstance(value, (pd.Timedelta, np.timedelta64, timedelta)):
        return _timedelta_label(value)
    if isinstance(value, pd.Period):
        return f"Period:{value}:{value.freqstr}"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{bool(value)}"
    if isinstance(value, (numbers.Number, Decimal)):
        return _numeric_label(value)
    if isinstance(value, str):
        return f"str:{value}"
    return f"{type(value).__name__}:{value}"


def _string_label(value):
    """Escape the missing sentinel without changing ordinary string labels."""
    if isinstance(value, str) and (value == "missing" or value.startswith("str:")):
        return f"str:{value}"
    return value


def _object_none_columns(X):
    """Retain actual None categories when later inputs use extension dtypes."""
    if isinstance(X, pd.DataFrame):
        return [column for column in X.select_dtypes(include="object") if any(value is None for value in X[column])]
    return ()


def _categorical_as_object(X, category_columns=None, string_columns=None, none_columns=()):
    """Represent pandas categorical values and missing markers for sklearn."""
    if isinstance(X, pd.DataFrame):
        values = X.astype(object)
        if category_columns is None:
            category_columns = X.select_dtypes(include="category").columns
        if string_columns is None:
            string_columns = [column for column in X if isinstance(X[column].dtype, pd.StringDtype)]
        for column in X:
            dtype = X[column].dtype
            use_labels = column in category_columns
            use_strings = column in string_columns
            if not use_labels and not use_strings:
                if isinstance(dtype, (pd.StringDtype, pd.CategoricalDtype)):
                    missing = None if column in none_columns else np.nan
                    values[column] = values[column].where(X[column].notna(), missing)
                continue
            if use_labels:
                values[column] = X[column].map(_categorical_label).astype(object)
            elif use_strings:
                values[column] = X[column].map(_string_label).astype(object)
            # Fill extension markers before the imputer can drop an empty column.
            values[column] = values[column].where(X[column].notna(), "missing")
        return values
    return X


class _CategoricalTransformer(FunctionTransformer):
    """Keep fitted category labels consistent when prediction dtypes change."""

    def fit(self, X, y=None):
        self.input_columns_ = X.columns if isinstance(X, pd.DataFrame) else None
        shape = np.shape(X)
        if len(shape) == 2:
            self.n_features_in_ = shape[1]
        if self.input_columns_ is not None and all(isinstance(column, str) for column in self.input_columns_):
            self.feature_names_in_ = np.asarray(self.input_columns_, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.category_columns_ = (
            X.select_dtypes(include="category").columns
            if isinstance(X, pd.DataFrame) else ()
        )
        self.string_columns_ = (
            [column for column in X if isinstance(X[column].dtype, pd.StringDtype)]
            if isinstance(X, pd.DataFrame) else ()
        )
        self.none_columns_ = _object_none_columns(X)
        self.nan_columns_ = [column for column in self.none_columns_
                             if any(missing and value is not None
                                    for value, missing in zip(X[column], X[column].isna()))]
        return super().fit(X, y)

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) and any((len(self.category_columns_), len(self.string_columns_),
                                                    len(self.none_columns_))):
            X = pd.DataFrame(X, columns=self.input_columns_, dtype=object)
            for column in self.none_columns_:
                if column not in self.nan_columns_:
                    X[column] = X[column].where(X[column].notna(), None)
        values = _categorical_as_object(X, self.category_columns_, self.string_columns_, self.none_columns_)
        return super().transform(values)

    def get_feature_names_out(self, input_features=None):
        if (
            hasattr(FunctionTransformer, "get_feature_names_out")
            and getattr(self, "feature_names_out", None) is not None
        ):
            return super().get_feature_names_out(input_features)
        return _check_feature_names_in(self, input_features)


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

    transformer_kwargs = {}
    if "feature_names_out" in inspect.signature(FunctionTransformer).parameters:
        transformer_kwargs["feature_names_out"] = "one-to-one"
    return Pipeline(
        steps=[
            ("to_object", _CategoricalTransformer(**transformer_kwargs)),
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
    categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns

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
