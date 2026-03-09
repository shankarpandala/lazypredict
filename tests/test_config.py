"""Tests for config module."""

from lazypredict.config import (
    DEFAULT_CARDINALITY_THRESHOLD,
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_STATE,
    REMOVED_CLASSIFIERS,
    REMOVED_REGRESSORS,
    VALID_ENCODERS,
)


def test_valid_encoders():
    assert "onehot" in VALID_ENCODERS
    assert "ordinal" in VALID_ENCODERS
    assert "target" in VALID_ENCODERS
    assert "binary" in VALID_ENCODERS


def test_removed_classifiers_is_frozenset():
    assert isinstance(REMOVED_CLASSIFIERS, frozenset)
    assert "ClassifierChain" in REMOVED_CLASSIFIERS


def test_removed_regressors_is_frozenset():
    assert isinstance(REMOVED_REGRESSORS, frozenset)
    assert "TheilSenRegressor" in REMOVED_REGRESSORS


def test_defaults():
    assert DEFAULT_RANDOM_STATE == 42
    assert DEFAULT_CARDINALITY_THRESHOLD == 11
    assert DEFAULT_N_JOBS == -1
