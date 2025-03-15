"""
GPU utilities for lazypredict.
"""
import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

logger = logging.getLogger("lazypredict.gpu")

# Dictionary mapping scikit-learn models to cuML equivalents
SKLEARN_TO_CUML_MAP = {
    'RandomForestClassifier': 'cuml.ensemble.RandomForestClassifier',
    'RandomForestRegressor': 'cuml.ensemble.RandomForestRegressor',
    'KNeighborsClassifier': 'cuml.neighbors.KNeighborsClassifier',
    'KNeighborsRegressor': 'cuml.neighbors.KNeighborsRegressor',
    'LogisticRegression': 'cuml.linear_model.LogisticRegression',
    'LinearRegression': 'cuml.linear_model.LinearRegression',
    'Ridge': 'cuml.linear_model.Ridge',
    'Lasso': 'cuml.linear_model.Lasso',
    'ElasticNet': 'cuml.linear_model.ElasticNet',
    'PCA': 'cuml.decomposition.PCA',
    'TSNE': 'cuml.manifold.TSNE',
    'DBSCAN': 'cuml.cluster.DBSCAN',
    'KMeans': 'cuml.cluster.KMeans',
}

def is_gpu_available() -> bool:
    """Check if GPU is available for computation.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise.
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

def get_gpu_model(model_name: str, fallback_to_cpu: bool = True) -> Optional[Type]:
    """Get a GPU-accelerated model if available.
    
    Parameters
    ----------
    model_name : str
        Name of the scikit-learn model.
    fallback_to_cpu : bool, default=True
        Whether to fall back to CPU if GPU model is not available.
        
    Returns
    -------
    Optional[Type]
        The model class or None if not available.
    """
    # Check if we have a GPU equivalent
    if model_name not in SKLEARN_TO_CUML_MAP:
        if fallback_to_cpu:
            return get_cpu_model(model_name)
        return None
    
    # Check if GPU is available
    if not is_gpu_available() or not is_cuml_available():
        if fallback_to_cpu:
            return get_cpu_model(model_name)
        return None
    
    # Try to import the GPU model
    try:
        module_path, class_name = SKLEARN_TO_CUML_MAP[model_name].rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import GPU model {model_name}: {e}")
        if fallback_to_cpu:
            return get_cpu_model(model_name)
        return None

def get_cpu_model(model_name: str) -> Optional[Type]:
    """Get a CPU model from scikit-learn.
    
    Parameters
    ----------
    model_name : str
        Name of the scikit-learn model.
        
    Returns
    -------
    Optional[Type]
        The model class or None if not available.
    """
    try:
        # Try different common scikit-learn module paths
        for module_path in [
            f"sklearn.ensemble.{model_name}",
            f"sklearn.linear_model.{model_name}",
            f"sklearn.neighbors.{model_name}",
            f"sklearn.tree.{model_name}",
            f"sklearn.svm.{model_name}",
            f"sklearn.naive_bayes.{model_name}",
            f"sklearn.discriminant_analysis.{model_name}",
            f"sklearn.neural_network.{model_name}",
        ]:
            try:
                module_path, class_name = module_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError):
                pass
                
        # If not found in common paths, try the all_estimators approach
        from sklearn.utils import all_estimators
        for name, estimator in all_estimators():
            if name == model_name:
                return estimator
                
        return None
    except Exception as e:
        logger.error(f"Error getting CPU model {model_name}: {e}")
        return None

def get_best_model(model_name: str, prefer_gpu: bool = True) -> Optional[Type]:
    """Get the best available model implementation (GPU or CPU).
    
    Parameters
    ----------
    model_name : str
        Name of the model.
    prefer_gpu : bool, default=True
        Whether to prefer GPU implementation if available.
        
    Returns
    -------
    Optional[Type]
        The model class or None if not available.
    """
    if prefer_gpu and is_gpu_available() and is_cuml_available():
        model_class = get_gpu_model(model_name)
        if model_class is not None:
            logger.info(f"Using GPU-accelerated version of {model_name}")
            return model_class
    
    # Fall back to CPU implementation
    model_class = get_cpu_model(model_name)
    if model_class is not None:
        logger.info(f"Using CPU version of {model_name}")
        return model_class
    
    logger.warning(f"Model {model_name} not found in either GPU or CPU implementations")
    return None 