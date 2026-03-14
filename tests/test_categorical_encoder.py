"""
Tests for categorical encoder options
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lazypredict.Supervised import LazyClassifier, LazyRegressor


def create_mixed_dataset():
    """Create a dataset with both numeric and categorical features"""
    # Generate numeric features
    X_num, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                                   n_redundant=1, random_state=42)
    X_df = pd.DataFrame(X_num, columns=[f'num_{i}' for i in range(5)])

    # Add low cardinality categorical features
    X_df['cat_low_1'] = np.random.choice(['A', 'B', 'C'], size=200)
    X_df['cat_low_2'] = np.random.choice(['X', 'Y'], size=200)

    # Add high cardinality categorical feature
    X_df['cat_high'] = [f'cat_{i % 50}' for i in range(200)]

    return X_df, y


def test_onehot_encoder():
    """Test one-hot encoder option"""
    X, y = create_mixed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        categorical_encoder='onehot',
        classifiers=[LogisticRegression]
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'Accuracy' in models.columns
    assert models['Accuracy'].iloc[0] > 0


def test_ordinal_encoder():
    """Test ordinal encoder option"""
    X, y = create_mixed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        categorical_encoder='ordinal',
        classifiers=[RandomForestClassifier]
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'Accuracy' in models.columns


def test_target_encoder_fallback():
    """Test target encoder with fallback when category_encoders not available"""
    X, y = create_mixed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # This should work even if category_encoders isn't installed (falls back to ordinal)
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        categorical_encoder='target',
        classifiers=[DecisionTreeClassifier]
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'Accuracy' in models.columns


def test_binary_encoder_fallback():
    """Test binary encoder with fallback when category_encoders not available"""
    X, y = create_mixed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # This should work even if category_encoders isn't installed (falls back to onehot)
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        categorical_encoder='binary',
        classifiers=[KNeighborsClassifier]
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'Accuracy' in models.columns


def test_invalid_encoder():
    """Test that invalid encoder raises error at init time"""
    with pytest.raises(ValueError, match="categorical_encoder must be one of"):
        LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            categorical_encoder='invalid_encoder',
            classifiers=[LogisticRegression]
        )


def test_regressor_ordinal_encoder():
    """Test ordinal encoder with LazyRegressor"""
    X, _ = create_mixed_dataset()
    y = np.random.randn(len(X))  # Continuous target for regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        categorical_encoder='ordinal',
        regressors=[LinearRegression, Ridge]
    )
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'R-Squared' in models.columns


def test_regressor_onehot_encoder():
    """Test one-hot encoder with LazyRegressor"""
    X, _ = create_mixed_dataset()
    y = np.random.randn(len(X))  # Continuous target for regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        categorical_encoder='onehot',
        regressors=[DecisionTreeRegressor]
    )
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'R-Squared' in models.columns


def test_default_encoder():
    """Test default encoder (should be onehot)"""
    X, y = create_mixed_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[LogisticRegression]
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert len(models) > 0
    assert 'Accuracy' in models.columns
    # Should work fine with default onehot encoder


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
