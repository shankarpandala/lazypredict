"""Explainability utilities for LazyPredict.

Provides model-agnostic feature importance via permutation importance (zero deps)
and optional SHAP-based explanations for tree/linear models.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger("lazypredict")

# Optional SHAP
try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def explain_permutation(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: Any,
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compute permutation importance for all fitted models.

    Parameters
    ----------
    models : dict
        Mapping of model name to fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : array-like
        Test target.
    n_repeats : int
        Number of permutation repeats.
    random_state : int
        Seed for reproducibility.
    n_jobs : int
        Parallel jobs for permutation importance.

    Returns
    -------
    pd.DataFrame
        DataFrame with features as rows, models as columns,
        values are mean importance scores.
    """
    if not models:
        raise ValueError("No fitted models. Call fit() first.")

    if isinstance(X_test, np.ndarray):
        cols = [f"feature_{i}" for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test, columns=cols)

    feature_names = list(X_test.columns)
    importance_dict: Dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        try:
            result = permutation_importance(
                pipe,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            importance_dict[name] = result.importances_mean
        except Exception as exc:
            logger.warning("Permutation importance failed for %s: %s", name, exc)
            importance_dict[name] = np.full(len(feature_names), np.nan)

    df = pd.DataFrame(importance_dict, index=feature_names)
    df.index.name = "Feature"
    return df


def _get_shap_explainer(pipe: Pipeline, X_background: np.ndarray, step_name: str):
    """Select the appropriate SHAP explainer for the inner model.

    Returns (explainer, X_transformed) or raises ImportError.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "shap is required for SHAP explanations. "
            "Install with: pip install lazypredict[explain]"
        )

    model = pipe.named_steps[step_name]
    preprocessor = pipe.named_steps.get("preprocessor")

    if preprocessor is not None:
        X_transformed = preprocessor.transform(X_background)
    else:
        X_transformed = X_background

    # Try TreeExplainer first (fastest)
    model_type = type(model).__name__
    tree_models = {
        "RandomForestClassifier", "RandomForestRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "AdaBoostClassifier", "AdaBoostRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
    }
    linear_models = {
        "LinearRegression", "Ridge", "RidgeCV", "Lasso", "LassoCV",
        "ElasticNet", "ElasticNetCV", "LogisticRegression",
        "SGDClassifier", "SGDRegressor", "BayesianRidge",
        "LinearSVC", "LinearSVR",
    }

    if model_type in tree_models:
        return shap.TreeExplainer(model), X_transformed
    elif model_type in linear_models:
        return shap.LinearExplainer(model, X_transformed), X_transformed
    else:
        # Fall back to permutation explainer (model-agnostic, faster than Kernel)
        return shap.PermutationExplainer(model.predict, X_transformed), X_transformed


def explain_shap(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    step_name: str = "regressor",
    max_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """Compute SHAP values for all fitted models.

    Parameters
    ----------
    models : dict
        Mapping of model name to fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test feature matrix.
    step_name : str
        Name of the estimator step in the Pipeline.
    max_samples : int
        Max samples for background/explanation (for speed).

    Returns
    -------
    dict
        Mapping of model name to SHAP values array (n_samples, n_features).
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "shap is required for SHAP explanations. "
            "Install with: pip install lazypredict[explain]"
        )

    if isinstance(X_test, np.ndarray):
        cols = [f"feature_{i}" for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test, columns=cols)

    X_sample = X_test.iloc[:max_samples] if len(X_test) > max_samples else X_test
    X_bg = X_test.values[:max_samples]

    shap_values: Dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        try:
            explainer, X_transformed = _get_shap_explainer(pipe, X_bg, step_name)
            X_explain = X_transformed[:len(X_sample)]
            sv = explainer.shap_values(X_explain)
            if isinstance(sv, list):
                sv = sv[0]  # binary classification: take class 1
            shap_values[name] = sv
        except Exception as exc:
            logger.warning("SHAP explanation failed for %s: %s", name, exc)

    return shap_values
