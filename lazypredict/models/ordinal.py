"""
Ordinal regression module for LazyPredict.
"""

from ..base import Lazy

class LazyOrdinalRegressor(Lazy):
    """
    Automated machine learning for ordinal regression.
    
    This class automates the process of training and evaluating multiple ordinal regression models.
    
    Parameters
    ----------
    verbose : int, default=1
        Verbosity level (0: no output, 1: minimal output, 2: detailed output)
    
    ignore_warnings : bool, default=True
        Whether to ignore warnings during model training
        
    custom_metric : callable, default=None
        Custom metric function (func(y_true, y_pred) should return a score)
        
    regressors : list, default=None
        List of regressor classes to use, if None all available regressors are used
        
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(
        self,
        verbose=1,
        ignore_warnings=True,
        custom_metric=None,
        regressors=None,
        random_state=42,
    ):
        super().__init__(verbose=verbose, ignore_warnings=ignore_warnings, random_state=random_state)
        self.custom_metric = custom_metric
        self.regressors = regressors
        self.models = {}
        
    def fit(self, X_train, X_test, y_train, y_test):
        """
        Fit all ordinal regression models.
        
        Parameters
        ----------
        X_train : array-like
            Training features
            
        X_test : array-like
            Test features
            
        y_train : array-like
            Training target
            
        y_test : array-like
            Test target
            
        Returns
        -------
        scores : pandas.DataFrame
            DataFrame with performance metrics for each model
            
        predictions : dict
            Dictionary with predictions for each model
        """
        raise NotImplementedError("LazyOrdinalRegressor.fit() is not implemented yet")
    
    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        Train and return all models.
        
        Parameters
        ----------
        X_train : array-like
            Training features
            
        X_test : array-like
            Test features
            
        y_train : array-like
            Training target
            
        y_test : array-like
            Test target
            
        Returns
        -------
        models : dict
            Dictionary with trained models
        """
        raise NotImplementedError("LazyOrdinalRegressor.provide_models() is not implemented yet") 