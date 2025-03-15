"""
Model registry for lazypredict.
"""
import logging
from typing import List, Type, Optional

logger = logging.getLogger("lazypredict.model_registry")

# Import scikit-learn models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDRegressor,
    BayesianRidge,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Define available models
REGRESSORS = [
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    DecisionTreeRegressor,
    KNeighborsRegressor,
    SVR,
    SGDRegressor,
    BayesianRidge,
    MLPRegressor,
    ExtraTreesRegressor,
]

CLASSIFIERS = [
    LogisticRegression,
    RidgeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    DecisionTreeClassifier,
    KNeighborsClassifier,
    SVC,
    SGDClassifier,
    GaussianNB,
    MLPClassifier,
    ExtraTreesClassifier,
]

def get_regression_models(models: Optional[List[str]] = None) -> List[Type]:
    """Get regression models.
    
    Parameters
    ----------
    models : List[str], optional (default=None)
        List of model names to include. If None, all models are included.
        
    Returns
    -------
    List[Type]
        List of regression model classes.
    """
    if models is None:
        return REGRESSORS
    
    filtered_models = []
    for model_name in models:
        model_name_str = str(model_name).lower()
        for model in REGRESSORS:
            if model.__name__.lower() == model_name_str:
                filtered_models.append(model)
                break
        else:
            logger.warning(f"Model {model_name} not found in regressor registry")
    
    return filtered_models if filtered_models else REGRESSORS

def get_classification_models(models: Optional[List[str]] = None) -> List[Type]:
    """Get classification models.
    
    Parameters
    ----------
    models : List[str], optional (default=None)
        List of model names to include. If None, all models are included.
        
    Returns
    -------
    List[Type]
        List of classification model classes.
    """
    if models is None:
        return CLASSIFIERS
    
    filtered_models = []
    for model_name in models:
        model_name_str = str(model_name).lower()
        for model in CLASSIFIERS:
            if model.__name__.lower() == model_name_str:
                filtered_models.append(model)
                break
        else:
            logger.warning(f"Model {model_name} not found in classifier registry")
    
    return filtered_models if filtered_models else CLASSIFIERS 