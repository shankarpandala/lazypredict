"""Model registry for lazypredict."""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger("lazypredict.model_registry")

# Default classifiers
_CLASSIFIERS = {
    "RandomForestClassifier": RandomForestClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "GaussianNB": GaussianNB,
    "AdaBoostClassifier": AdaBoostClassifier,
    "SVC": SVC,
    "SGDClassifier": SGDClassifier,
}

# Default regressors
_REGRESSORS = {
    "RandomForestRegressor": RandomForestRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "KNeighborsRegressor": KNeighborsRegressor,
    "SVR": SVR,
    "SGDRegressor": SGDRegressor,
    "AdaBoostRegressor": AdaBoostRegressor,
}

# Runtime registries
CLASSIFIERS = dict(_CLASSIFIERS)
REGRESSORS = dict(_REGRESSORS)


def _get_class_name(model_class: Union[Type[BaseEstimator], str]) -> str:
    """Get the name of a model class."""
    if isinstance(model_class, str):
        return model_class
    return model_class.__name__


def get_classification_models(
    names: Optional[List[Union[str, Type[BaseEstimator]]]] = None,
) -> List[Type[BaseEstimator]]:
    """Get classification models.

    Parameters
    ----------
    names : list of str or model classes, optional (default=None)
        List of model names or classes to include. If None, all models are included.

    Returns
    -------
    list
        List of model classes
    """
    if not names:
        return list(CLASSIFIERS.values())

    models = []
    for name in names:
        if isinstance(name, type):
            # It's already a class, add it directly
            models.append(name)
            # Register it if not already in the registry
            class_name = _get_class_name(name)
            if class_name not in CLASSIFIERS:
                register_classifier(class_name, name)
        elif isinstance(name, str):
            # It's a string name, look up in registry
            if name in CLASSIFIERS:
                models.append(CLASSIFIERS[name])
            else:
                # Try to find in sklearn
                try:
                    from sklearn.utils import all_estimators

                    classifiers = dict(all_estimators(type_filter="classifier"))
                    if name in classifiers:
                        model_class = classifiers[name]
                        register_classifier(name, model_class)
                        models.append(model_class)
                    else:
                        logger.warning(f"Model {name} not found in classifier registry")
                except Exception as e:
                    logger.warning(f"Error finding model {name}: {str(e)}")
        else:
            logger.warning(f"Unexpected model type: {type(name)}")

    # Return what we have, even if some models weren't found
    return models


def get_regression_models(
    names: Optional[List[Union[str, Type[BaseEstimator]]]] = None,
) -> List[Type[BaseEstimator]]:
    """Get regression models.

    Parameters
    ----------
    names : list of str or model classes, optional (default=None)
        List of model names or classes to include. If None, all models are included.

    Returns
    -------
    list
        List of model classes
    """
    if not names:
        return list(REGRESSORS.values())

    models = []
    for name in names:
        if isinstance(name, type):
            # It's already a class, add it directly
            models.append(name)
            # Register it if not already in the registry
            class_name = _get_class_name(name)
            if class_name not in REGRESSORS:
                register_regressor(class_name, name)
        elif isinstance(name, str):
            # It's a string name, look up in registry
            if name in REGRESSORS:
                models.append(REGRESSORS[name])
            else:
                # Try to find in sklearn
                try:
                    from sklearn.utils import all_estimators

                    regressors = dict(all_estimators(type_filter="regressor"))
                    if name in regressors:
                        model_class = regressors[name]
                        register_regressor(name, model_class)
                        models.append(model_class)
                    else:
                        logger.warning(f"Model {name} not found in regressor registry")
                except Exception as e:
                    logger.warning(f"Error finding model {name}: {str(e)}")
        else:
            logger.warning(f"Unexpected model type: {type(name)}")

    # Return what we have, even if some models weren't found
    return models


def register_classifier(name: str, model_class: Type[BaseEstimator]) -> None:
    """Register a new classifier.

    Parameters
    ----------
    name : str
        Name of the classifier
    model_class : type
        Classifier class to register
    """
    if isinstance(model_class, type):
        CLASSIFIERS[name] = model_class
    else:
        logger.warning(f"Cannot register non-class type: {type(model_class)}")


def register_regressor(name: str, model_class: Type[BaseEstimator]) -> None:
    """Register a new regressor.

    Parameters
    ----------
    name : str
        Name of the regressor
    model_class : type
        Regressor class to register
    """
    if isinstance(model_class, type):
        REGRESSORS[name] = model_class
    else:
        logger.warning(f"Cannot register non-class type: {type(model_class)}")


def filter_models(
    models: Dict[str, Type[BaseEstimator]], exclude: Optional[List[str]] = None
) -> Dict[str, Type[BaseEstimator]]:
    """Filter models based on exclusion list.

    Parameters
    ----------
    models : dict
        Dictionary of model name to model class
    exclude : list of str, optional (default=None)
        List of model names to exclude

    Returns
    -------
    dict
        Filtered dictionary of models
    """
    if exclude is None:
        return models

    return {name: model for name, model in models.items() if name not in exclude}


def reset_registry():
    """Reset registries to default state."""
    global CLASSIFIERS, REGRESSORS
    CLASSIFIERS = dict(_CLASSIFIERS)
    REGRESSORS = dict(_REGRESSORS)
