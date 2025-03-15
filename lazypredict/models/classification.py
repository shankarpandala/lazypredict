"""
Classification module for lazypredict.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from ..utils import (
    configure_mlflow,
    start_run,
    end_run,
    log_params,
    log_model_performance,
    log_dataframe,
    get_model_name,
)
from ..utils.preprocessing import create_preprocessor
from ..utils.gpu import get_best_model

logger = logging.getLogger("lazypredict.classification")

# Define excluded scikit-learn classifiers (e.g., those with specific requirements)
EXCLUDED_CLASSIFIERS = [
    "ClassifierChain",
    "ComplementNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "MLPClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
    "StackingClassifier",
    "CalibratedClassifierCV",
]

class LazyClassifier:
    """Automated classification model selection and evaluation.
    
    This class automates the process of training and evaluating multiple classification models.
    It provides a simple interface to quickly compare different models on a given dataset.
    
    Parameters
    ----------
    verbose : int, optional (default=0)
        Verbosity level. Higher values print more information.
    ignore_warnings : bool, optional (default=True)
        Whether to ignore warning messages.
    custom_metric : callable, optional (default=None)
        Custom scoring metric that takes y_true and y_pred as inputs.
    random_state : int, optional (default=None)
        Random state for reproducibility.
    classifiers : List[str], optional (default=None)
        List of classifier names to include. If None, all available classifiers will be used.
    """
    
    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        random_state: Optional[int] = None,
        classifiers: Optional[List[str]] = None,
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.random_state = random_state
        self.classifiers = classifiers
        self.progress_bar = None
        self.models = []
        self.predictions = {}
        self.scores = {}
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Import classification models
        from ..models.model_registry import get_classification_models
        
        # Filter models if needed
        self._filtered_models = get_classification_models(self.classifiers)

    def _check_data(
        self, 
        X_train: Union[pd.DataFrame, np.ndarray], 
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Check input data and convert to appropriate format.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        y_train : array-like of shape (n_samples,)
            Training labels.
        y_test : array-like of shape (n_samples,), optional (default=None)
            Test labels.
            
        Returns
        -------
        Tuple containing processed input data.
        """
        # Convert pandas DataFrames to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
            
        # Convert pandas Series to numpy arrays
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy()
        if y_test is not None and isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_test = y_test.to_numpy()
            
        # Reshape y if needed
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        if y_test is not None and len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
            
        return X_train, X_test, y_train, y_test
        
    def _get_classifiers(self) -> List[Tuple[str, ClassifierMixin]]:
        """Get the list of classifier models to use.
        
        Returns
        -------
        List[Tuple[str, ClassifierMixin]]
            List of (name, model) tuples.
        """
        try:
            from ..models.model_registry import get_classification_models
            return [(get_model_name(clf), clf) for clf in self._filtered_models]
        except Exception as e:
            self.logger.error(f"Error getting classifiers: {e}")
            return []
        
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        mlflow_tracking_uri: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, np.ndarray]]]:
        """Fit all classification algorithms and return performance metrics.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        y_train : array-like of shape (n_samples,)
            Training labels.
        y_test : array-like of shape (n_samples,)
            Test labels.
        mlflow_tracking_uri : str, optional (default=None)
            MLflow tracking URI. If None, will use the global tracking URI.
            
        Returns
        -------
        Tuple[pd.DataFrame, Optional[Dict[str, np.ndarray]]]
            DataFrame with performance metrics for each model and optionally a dictionary of predictions.
        """
        # Configure MLflow if tracking URI is provided
        if mlflow_tracking_uri:
            configure_mlflow(mlflow_tracking_uri)
            start_run(run_name="LazyClassifier")
            
            # Log run parameters
            run_params = {
                "verbose": self.verbose,
                "ignore_warnings": self.ignore_warnings,
                "custom_metric": self.custom_metric.__name__ if self.custom_metric else None,
                "random_state": self.random_state,
            }
            log_params(run_params)
        
        # Check input data
        X_train_np, X_test_np, y_train_np, y_test_np = self._check_data(X_train, X_test, y_train, y_test)
        
        # Create the preprocessor
        preprocessor = create_preprocessor(X_train_np)
        
        # Initialize storage
        scores_dict = {}
        predictions = {} if self.predictions else None
        custom_metric_values = {}
        
        # Get classifiers to use
        classifiers = self._get_classifiers()
        
        # Fit each classifier
        for name, classifier in tqdm(classifiers, desc="Fitting classifiers"):
            start_time = time.time()
            try:
                # Check if GPU model is available
                if self.random_state:
                    model_class = get_best_model(name, prefer_gpu=True)
                    if model_class is not None:
                        classifier = model_class
                
                # Create and fit the pipeline
                if hasattr(classifier, "random_state") or "random_state" in classifier().get_params():
                    clf = classifier(random_state=self.random_state)
                else:
                    clf = classifier()
                
                pipe = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", clf),
                    ]
                )
                
                # Fit the model
                pipe.fit(X_train_np, y_train_np)
                
                # Make predictions
                y_pred = pipe.predict(X_test_np)
                if self.predictions:
                    predictions[name] = y_pred
                
                # Calculate metrics
                metrics = {}
                metrics["Accuracy"] = accuracy_score(y_test_np, y_pred, normalize=True)
                metrics["Balanced Accuracy"] = balanced_accuracy_score(y_test_np, y_pred)
                metrics["F1 Score"] = f1_score(y_test_np, y_pred, average="weighted")
                
                # Calculate ROC AUC if possible (requires predict_proba)
                if hasattr(pipe, "predict_proba"):
                    try:
                        if len(np.unique(y_test_np)) > 1:  # Only calculate for non-degenerate cases
                            y_proba = pipe.predict_proba(X_test_np)
                            if y_proba.shape[1] == 2:  # Binary case
                                metrics["ROC AUC"] = roc_auc_score(y_test_np, y_proba[:, 1])
                            else:  # Multi-class case
                                metrics["ROC AUC"] = roc_auc_score(
                                    y_test_np, y_proba, multi_class="ovr", average="weighted"
                                )
                    except Exception as e:
                        if not self.ignore_warnings:
                            logger.warning(f"Error calculating ROC AUC for {name}: {str(e)}")
                        metrics["ROC AUC"] = float("nan")
                else:
                    metrics["ROC AUC"] = float("nan")
                
                # Add time taken
                metrics["Time taken"] = time.time() - start_time
                
                # Add custom metric if provided
                if self.custom_metric:
                    try:
                        custom_val = self.custom_metric(y_test_np, y_pred)
                        metrics["Custom Metric"] = custom_val
                        custom_metric_values[name] = custom_val
                    except Exception as e:
                        if not self.ignore_warnings:
                            logger.warning(f"Error calculating custom metric for {name}: {str(e)}")
                        metrics["Custom Metric"] = float("nan")
                
                # Store model and scores
                self.models.append(pipe)
                scores_dict[name] = metrics
                
                # Log model performance to MLflow if configured
                if mlflow_tracking_uri:
                    log_model_performance(name, metrics)
                
                if self.verbose > 0:
                    print(f"Model: {name}, Metrics: {metrics}")
                
            except Exception as e:
                if not self.ignore_warnings:
                    logger.warning(f"Error fitting {name}: {str(e)}")
        
        # Convert scores to DataFrame
        if not scores_dict:
            return pd.DataFrame(), predictions
        
        # Create DataFrame from scores
        scores_df = pd.DataFrame.from_dict(scores_dict, orient='index').reset_index()
        scores_df.rename(columns={'index': 'Model'}, inplace=True)
        
        # Sort by accuracy
        if 'Accuracy' in scores_df.columns:
            scores_df = scores_df.sort_values("Accuracy", ascending=False)
        
        # Log the full results table to MLflow
        if mlflow_tracking_uri:
            log_dataframe(scores_df, "model_comparison")
            end_run()
        
        return scores_df, predictions
        
    def provide_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Pipeline]:
        """Train and provide fitted pipeline objects for all models.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        y_train : array-like of shape (n_samples,)
            Training labels.
        y_test : array-like of shape (n_samples,)
            Test labels.
            
        Returns
        -------
        Dict[str, Pipeline]
            Dictionary of fitted pipeline objects for all models.
        """
        # If models haven't been trained yet, train them
        if not self.models:
            self.fit(X_train, X_test, y_train, y_test)
            
        return self.models
    
    def fit_optimize(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        estimator: Any,
        param_dist: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        cv: int = 5,
        scoring: Optional[Union[str, Callable]] = None,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Perform hyperparameter optimization using Optuna.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Training labels.
        estimator : estimator instance
            Estimator to optimize.
        param_dist : dict, optional (default=None)
            Parameter distributions for sampling.
            If None, default param_dist for the estimator will be used.
        n_trials : int, optional (default=100)
            Number of trials for hyperparameter optimization.
        cv : int, optional (default=5)
            Number of cross-validation folds.
        scoring : str or callable, optional (default=None)
            Scoring metric to use for hyperparameter optimization.
            If None, 'accuracy' will be used for classifiers.
        n_jobs : int, optional (default=-1)
            Number of jobs to run in parallel.
            -1 means using all processors.
            
        Returns
        -------
        Dict[str, Any]
            Best hyperparameters found.
        """
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.error("Optuna not installed. Please install optuna to use this function.")
            return {}
            
        # Check input data
        X_train, _, y_train, _ = self._check_data(X_train, X_train, y_train, y_train)
        
        # Set default scoring
        if scoring is None:
            scoring = "accuracy"
            
        # Set default param_dist if not provided
        if param_dist is None:
            if hasattr(estimator, "__name__"):
                estimator_name = estimator.__name__
            else:
                estimator_name = estimator.__class__.__name__
                
            # Define default param_dist based on estimator name
            if estimator_name in ["RandomForestClassifier", "ExtraTreesClassifier"]:
                param_dist = {
                    "n_estimators": optuna.distributions.IntDistribution(50, 1000),
                    "max_depth": optuna.distributions.IntDistribution(3, 50),
                    "min_samples_split": optuna.distributions.IntDistribution(2, 20),
                    "min_samples_leaf": optuna.distributions.IntDistribution(1, 20),
                    "max_features": optuna.distributions.CategoricalDistribution(["auto", "sqrt", "log2"]),
                }
            elif estimator_name == "XGBClassifier":
                param_dist = {
                    "n_estimators": optuna.distributions.IntDistribution(50, 1000),
                    "max_depth": optuna.distributions.IntDistribution(3, 50),
                    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
                    "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
                    "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
                }
            else:
                logger.warning(f"No default param_dist for {estimator_name}. Using empty param_dist.")
                param_dist = {}
        
        # Create preprocessor
        preprocessor = create_preprocessor(X_train)
        
        # Define objective function
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, distribution in param_dist.items():
                if isinstance(distribution, optuna.distributions.IntDistribution):
                    params[param_name] = trial.suggest_int(param_name, distribution.low, distribution.high)
                elif isinstance(distribution, optuna.distributions.FloatDistribution):
                    params[param_name] = trial.suggest_float(param_name, distribution.low, distribution.high)
                elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                    params[param_name] = trial.suggest_categorical(param_name, distribution.choices)
                    
            # Create model with sampled params
            model = estimator(**params)
            
            # Create pipeline
            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )
            
            # Evaluate model
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
            return scores.mean()
            
        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best params
        best_params = study.best_params
        
        # Print best params if verbose
        if self.verbose > 0:
            print(f"Best parameters: {best_params}")
            print(f"Best score: {study.best_value:.4f}")
            
        return best_params 