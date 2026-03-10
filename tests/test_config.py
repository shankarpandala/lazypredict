"""Tests for config module."""

from unittest.mock import MagicMock, patch

from lazypredict.config import (
    DEFAULT_CARDINALITY_THRESHOLD,
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_STATE,
    REMOVED_CLASSIFIERS,
    REMOVED_REGRESSORS,
    VALID_ENCODERS,
    get_cuml_models,
    get_gpu_model_params,
    is_gpu_available,
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


# ---------------------------------------------------------------------------
# GPU support tests
# ---------------------------------------------------------------------------


def test_get_gpu_model_params_use_gpu_false():
    """When use_gpu=False, always returns empty dict."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "xgboost.sklearn"
    assert get_gpu_model_params(mock_cls, use_gpu=False) == {}


def test_get_gpu_model_params_xgboost():
    """XGBoost should get device='cuda' when GPU is available."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "xgboost.sklearn"
    mock_cls.__name__ = "XGBClassifier"

    with patch("lazypredict.config.is_gpu_available", return_value=True):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {"device": "cuda"}


def test_get_gpu_model_params_lightgbm():
    """LightGBM should get device='gpu' when GPU is available."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "lightgbm.sklearn"
    mock_cls.__name__ = "LGBMClassifier"

    with patch("lazypredict.config.is_gpu_available", return_value=True):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {"device": "gpu"}


def test_get_gpu_model_params_catboost():
    """CatBoost should get task_type='GPU' when GPU is available."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "catboost.core"
    mock_cls.__name__ = "CatBoostClassifier"

    with patch("lazypredict.config.is_gpu_available", return_value=True):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {"task_type": "GPU"}


def test_get_gpu_model_params_catboost_no_gpu():
    """CatBoost should fallback to empty dict when no GPU."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "catboost.core"
    mock_cls.__name__ = "CatBoostClassifier"

    with patch("lazypredict.config.is_gpu_available", return_value=False):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {}


def test_get_gpu_model_params_cuml():
    """cuML models should return empty dict (GPU-native, no extra params)."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "cuml.linear_model.logistic_regression"
    mock_cls.__name__ = "LogisticRegression"

    with patch("lazypredict.config.is_gpu_available", return_value=True):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {}


def test_get_gpu_model_params_cuml_no_gpu():
    """cuML without GPU should warn and return empty dict."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "cuml.linear_model.logistic_regression"
    mock_cls.__name__ = "LogisticRegression"

    with patch("lazypredict.config.is_gpu_available", return_value=False):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {}


def test_get_gpu_model_params_unknown_model():
    """Unknown models should return empty dict."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "sklearn.linear_model"
    mock_cls.__name__ = "LinearRegression"

    result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {}


def test_get_gpu_model_params_xgboost_no_gpu():
    """XGBoost should fallback to empty dict when no GPU."""
    mock_cls = MagicMock()
    mock_cls.__module__ = "xgboost.sklearn"
    mock_cls.__name__ = "XGBRegressor"

    with patch("lazypredict.config.is_gpu_available", return_value=False):
        result = get_gpu_model_params(mock_cls, use_gpu=True)
    assert result == {}


def test_get_cuml_models_not_installed():
    """get_cuml_models returns empty dict when cuML is not installed."""
    # cuML is unlikely to be installed in test environments
    result = get_cuml_models()
    assert isinstance(result, dict)


def test_is_gpu_available_returns_bool():
    """is_gpu_available should return a boolean."""
    result = is_gpu_available()
    assert isinstance(result, bool)
