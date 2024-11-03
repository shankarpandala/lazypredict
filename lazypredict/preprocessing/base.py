# lazypredict/preprocessing/base.py

from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .feature_engineering import FeatureEngineer
from .feature_selection import FeatureSelector
from .transformers import CustomTransformer
from ..utils.backend import Backend
from ..utils.data_utils import get_card_split
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Preprocessor class that handles data preprocessing steps.
    """

    def __init__(
        self,
        feature_engineer: Optional[FeatureEngineer] = None,
        feature_selector: Optional[FeatureSelector] = None,
        transformers: Optional[List[CustomTransformer]] = None,
        use_gpu: bool = False,
    ):
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.transformers = transformers or []
        self.use_gpu = use_gpu
        self.backend = Backend.get_backend()
        self.pipeline = None
        self.logger = logger

    def build_preprocessor(self, sample_data) -> Pipeline:
        """
        Builds the preprocessing pipeline.
        """
        # Get the appropriate DataFrame type from the backend
        DataFrame = self.backend.DataFrame

        numeric_features = sample_data.select_dtypes(include=['number']).columns.tolist()
        categorical_features = sample_data.select_dtypes(exclude=['number']).columns.tolist()

        categorical_low_cardinality, categorical_high_cardinality = get_card_split(
            sample_data, categorical_features
        )

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler(with_mean=(not self.backend.is_gpu_backend))),
        ])

        categorical_transformer_low = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])

        categorical_transformer_high = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ])

        transformers = [
            ('numeric', numeric_transformer, numeric_features),
            ('categorical_low', categorical_transformer_low, categorical_low_cardinality),
            ('categorical_high', categorical_transformer_high, categorical_high_cardinality),
        ]

        # Add custom transformers
        for transformer in self.transformers:
            transformers.append(transformer.get_transformer())

        preprocessor = ColumnTransformer(transformers=transformers)

        # Add feature engineering and selection steps if provided
        steps = []

        if self.feature_engineer is not None:
            steps.append(('feature_engineering', self.feature_engineer))

        steps.append(('preprocessor', preprocessor))

        if self.feature_selector is not None:
            steps.append(('feature_selection', self.feature_selector))

        pipeline = Pipeline(steps=steps)

        return pipeline

    def fit(self, X, y=None):
        """
        Fits the preprocessor to the data.
        """
        # Fit feature engineer if provided
        if self.feature_engineer is not None:
            self.feature_engineer.fit(X, y)

        # Build and fit preprocessor pipeline
        self.pipeline = self.build_preprocessor(X)
        self.pipeline.fit(X, y)

    def transform(self, X):
        """
        Transforms the input data using the fitted preprocessor.
        """
        return self.pipeline.transform(X)
