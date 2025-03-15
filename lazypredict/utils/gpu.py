"""
GPU utilities for lazypredict.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger("lazypredict.gpu")

def is_gpu_available() -> bool:
    """Check if a GPU is available.
    
    Returns
    -------
    bool
        True if a GPU is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("torch not installed. GPU availability check failed.")
        return False

def is_cuml_available() -> bool:
    """Check if cuML is available.
    
    Returns
    -------
    bool
        True if cuML is available, False otherwise.
    """
    try:
        import cuml
        return True
    except ImportError:
        logger.warning("cuML not installed. GPU acceleration for ML models not available.")
        return False

def get_cpu_model(model_name: str) -> Optional[Type]:
    """Get CPU implementation of a model.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
        
    Returns
    -------
    Type or None
        CPU implementation of the model, or None if not found.
    """
    # Standard sklearn models
    sklearn_models = {
        # Linear models
        "LinearRegression": "sklearn.linear_model",
        "Ridge": "sklearn.linear_model",
        "Lasso": "sklearn.linear_model",
        "ElasticNet": "sklearn.linear_model",
        "SGDRegressor": "sklearn.linear_model",
        "SGDClassifier": "sklearn.linear_model",
        "LogisticRegression": "sklearn.linear_model",
        "RidgeClassifier": "sklearn.linear_model",
        
        # Ensemble models
        "RandomForestRegressor": "sklearn.ensemble",
        "RandomForestClassifier": "sklearn.ensemble",
        "GradientBoostingRegressor": "sklearn.ensemble",
        "GradientBoostingClassifier": "sklearn.ensemble",
        "AdaBoostRegressor": "sklearn.ensemble",
        "AdaBoostClassifier": "sklearn.ensemble",
        "ExtraTreesRegressor": "sklearn.ensemble",
        "ExtraTreesClassifier": "sklearn.ensemble",
        
        # SVM models
        "SVR": "sklearn.svm",
        "SVC": "sklearn.svm",
        
        # Neighbors models
        "KNeighborsRegressor": "sklearn.neighbors",
        "KNeighborsClassifier": "sklearn.neighbors",
        
        # Tree models
        "DecisionTreeRegressor": "sklearn.tree",
        "DecisionTreeClassifier": "sklearn.tree",
        
        # Neural networks
        "MLPRegressor": "sklearn.neural_network",
        "MLPClassifier": "sklearn.neural_network",
        
        # Naive Bayes
        "GaussianNB": "sklearn.naive_bayes",
    }
    
    # External models 
    if model_name == "XGBRegressor" or model_name == "XGBClassifier":
        try:
            import xgboost
            return getattr(xgboost, model_name)
        except (ImportError, AttributeError):
            return None
    elif model_name == "LGBMRegressor" or model_name == "LGBMClassifier":
        try:
            import lightgbm
            return getattr(lightgbm, model_name)
        except (ImportError, AttributeError):
            return None
    
    # Standard sklearn models
    if model_name in sklearn_models:
        module_name = sklearn_models[model_name]
        try:
            module = __import__(module_name, fromlist=[model_name])
            return getattr(module, model_name)
        except (ImportError, AttributeError):
            return None
            
    return None

def get_gpu_model(model_name: str) -> Optional[Type]:
    """Get GPU implementation of a model if available.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
        
    Returns
    -------
    Type or None
        GPU implementation of the model, or None if not found.
    """
    if not is_gpu_available() or not is_cuml_available():
        return None
        
    # Map sklearn model names to cuML
    try:
        import cuml
        
        cuml_models = {
            # Linear models
            "LinearRegression": cuml.LinearRegression,
            "Ridge": cuml.Ridge,
            "Lasso": cuml.Lasso,
            "ElasticNet": cuml.ElasticNet,
            
            # Ensemble models
            "RandomForestRegressor": cuml.ensemble.RandomForestRegressor,
            "RandomForestClassifier": cuml.ensemble.RandomForestClassifier,
            
            # SVM models
            "SVR": cuml.svm.SVR,
            "SVC": cuml.svm.SVC,
            
            # Neighbors models
            "KNeighborsRegressor": cuml.neighbors.KNeighborsRegressor,
            "KNeighborsClassifier": cuml.neighbors.KNeighborsClassifier,
        }
        
        if model_name in cuml_models:
            return cuml_models[model_name]
            
    except (ImportError, AttributeError):
        pass
        
    # For external models like XGBoost and LightGBM
    # They support GPU natively with parameters
    if model_name in ["XGBRegressor", "XGBClassifier", "LGBMRegressor", "LGBMClassifier"]:
        return get_cpu_model(model_name)  # These models support GPU natively
        
    return None

def get_best_model(model_name: str, prefer_gpu: bool = True) -> Optional[Type]:
    """Get the best implementation of a model (GPU or CPU).
    
    Parameters
    ----------
    model_name : str
        Name of the model.
    prefer_gpu : bool, optional (default=True)
        Whether to prefer GPU implementation when available.
        
    Returns
    -------
    Type or None
        Best implementation of the model, or None if not found.
    """
    if prefer_gpu:
        gpu_model = get_gpu_model(model_name)
        if gpu_model is not None:
            return gpu_model
            
    cpu_model = get_cpu_model(model_name)
    if cpu_model is not None:
        return cpu_model
        
    logger.warning(f"Model {model_name} not found in either GPU or CPU implementations")
    return None 