"""
Regression module for LazyPredict.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Lazy import of sklearn modules
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.svm

from ..base import Lazy
from ..utils import get_model_name
from ..metrics import get_regression_metrics

# Configure logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Define available regressors
REGRESSORS = [
    sklearn.linear_model.LinearRegression,
    sklearn.linear_model.Ridge,
    sklearn.linear_model.Lasso,
    sklearn.linear_model.ElasticNet,
    sklearn.ensemble.RandomForestRegressor,
    sklearn.ensemble.GradientBoostingRegressor,
    sklearn.ensemble.AdaBoostRegressor,
    sklearn.tree.DecisionTreeRegressor,
    sklearn.neighbors.KNeighborsRegressor,
    sklearn.svm.SVR,
]

class LazyRegressor(Lazy):
    """
    Automated machine learning for regression.
    
    This class automates the process of training and evaluating multiple regression models.
    
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
        self.regressors = regressors if regressors is not None else REGRESSORS
        self.models = {}
        
    def fit(self, X_train, X_test, y_train, y_test):
        """
        Fit all regression models.
        
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
        # Convert data to numpy arrays if needed
        X_train, X_test, y_train, y_test = self._check_data(X_train, X_test, y_train, y_test)
        
        # Initialize dictionaries for results
        scores_dict = {}
        predictions = {}
        
        # Train and evaluate each model
        for Model in tqdm(self.regressors, desc="Fitting regressors"):
            model_name = get_model_name(Model)
            
            try:
                # Initialize and train model
                start_time = time.time()
                model = Model(random_state=self.random_state) if hasattr(Model, 'random_state') else Model()
                model.fit(X_train, y_train)
                
                # Get predictions
                y_pred = model.predict(X_test)
                predictions[model_name] = y_pred
                
                # Calculate metrics
                metrics = get_regression_metrics(y_test, y_pred)
                
                # Add custom metric if provided
                if self.custom_metric:
                    metrics['Custom Metric'] = self.custom_metric(y_test, y_pred)
                
                # Add time taken
                metrics['Time taken'] = time.time() - start_time
                
                # Store model and scores
                self.models[model_name] = model
                scores_dict[model_name] = metrics
                
                if self.verbose > 0:
                    print({"Model": model_name, **metrics})
                
            except Exception as e:
                if self.verbose > 0:
                    logger.warning(f"Error fitting {model_name}: {str(e)}")
        
        # Convert scores to DataFrame
        if not scores_dict:
            return pd.DataFrame(), {}
        
        # Convert the dictionary into a DataFrame with the model name as a column
        scores_df = pd.DataFrame.from_dict(scores_dict, orient='index').reset_index()
        scores_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # Sort by R-Squared value
        scores_df = scores_df.sort_values("R-Squared", ascending=False)
        
        return scores_df, predictions
    
    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        Train and return all models.
        
        This method fits all models and returns them without calculating metrics.
        
        Parameters
        ----------
        X_train : array-like
            Training features
            
        X_test : array-like
            Test features (not used in this method but kept for API consistency)
            
        y_train : array-like
            Training target
            
        y_test : array-like
            Test target (not used in this method but kept for API consistency)
            
        Returns
        -------
        models : dict
            Dictionary with trained models
        """
        # Clear existing models
        self.models = {}
        
        # Convert data to numpy arrays if needed
        X_train, _, y_train, _ = self._check_data(X_train, X_test, y_train, y_test)
        
        # Train each model
        for Model in tqdm(self.regressors, desc="Training models"):
            model_name = get_model_name(Model)
            
            try:
                # Initialize and train model
                model = Model(random_state=self.random_state) if hasattr(Model, 'random_state') else Model()
                model.fit(X_train, y_train)
                
                # Store model
                self.models[model_name] = model
                
            except Exception as e:
                if self.verbose > 0:
                    logger.warning(f"Error training {model_name}: {str(e)}")
        
        return self.models 