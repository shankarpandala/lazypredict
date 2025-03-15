"""
Classification models for lazypredict.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    BaseLazy,
    create_preprocessor,
    get_best_model,
    log_model_performance,
    log_params,
)

logger = logging.getLogger("lazypredict.classification")

# Default classifiers to exclude
EXCLUDED_CLASSIFIERS = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
]

class LazyClassifier(BaseLazy):
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    
    Parameters
    ----------
    verbose : int, optional (default=0)
        Controls the verbosity of the LazyClassifier.
        0 = no output
        1 = basic output
        2 = detailed output
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorithms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : list or str, optional (default="all")
        When "all", uses all classifiers. When list is provided, uses only the specified classifiers.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    use_gpu : bool, optional (default=True)
        Whether to use GPU acceleration when available.
    
    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
    | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
    | MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        classifiers: Union[str, List] = "all",
        use_gpu: bool = True,
    ):
        """Initialize the LazyClassifier."""
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
        )
        self.classifiers = classifiers
        self.use_gpu = use_gpu
        self._estimator_type = "classifier"
        
    def _get_classifiers(self) -> List[Tuple[str, ClassifierMixin]]:
        """Get classifiers to use.
        
        Returns
        -------
        List[Tuple[str, ClassifierMixin]]
            List of (name, classifier) tuples.
        """
        # Import optional dependencies here to avoid import errors
        try:
            import xgboost
            xgb_available = True
        except ImportError:
            xgb_available = False
            logger.warning("xgboost not installed, XGBClassifier will not be used.")
            
        try:
            import lightgbm
            lgbm_available = True
        except ImportError:
            lgbm_available = False
            logger.warning("lightgbm not installed, LGBMClassifier will not be used.")
            
        # Get all scikit-learn classifiers
        from sklearn.utils import all_estimators
        classifiers = [
            est for est in all_estimators()
            if (issubclass(est[1], ClassifierMixin) and est[0] not in EXCLUDED_CLASSIFIERS)
        ]
        
        # Add XGBoost and LightGBM if available
        if xgb_available:
            classifiers.append(("XGBClassifier", xgboost.XGBClassifier))
        if lgbm_available:
            classifiers.append(("LGBMClassifier", lightgbm.LGBMClassifier))
            
        # Filter if specific classifiers are requested
        if self.classifiers != "all":
            if not isinstance(self.classifiers, list):
                raise ValueError("classifiers must be 'all' or a list of classifiers")
            
            # Extract classifier names if custom classes are provided
            if all(isinstance(clf, type) for clf in self.classifiers):
                classifiers = [(clf.__name__, clf) for clf in self.classifiers]
            else:
                # Filter by name
                filtered_classifiers = []
                for name, classifier in classifiers:
                    if name in self.classifiers:
                        filtered_classifiers.append((name, classifier))
                classifiers = filtered_classifiers
        
        return classifiers
        
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
            from ..utils import configure_mlflow, start_run, end_run
            configure_mlflow(mlflow_tracking_uri)
            start_run(run_name="LazyClassifier")
            
            # Log run parameters
            run_params = {
                "verbose": self.verbose,
                "ignore_warnings": self.ignore_warnings,
                "custom_metric": self.custom_metric.__name__ if self.custom_metric else None,
                "predictions": self.predictions,
                "random_state": self.random_state,
                "use_gpu": self.use_gpu,
            }
            log_params(run_params)
        
        # Check input data
        X_train, X_test, y_train, y_test = self._check_data(X_train, X_test, y_train, y_test)
        
        # Initialize metrics lists
        names = []
        accuracy = []
        balanced_accuracy = []
        roc_auc = []
        f1 = []
        time_taken = []
        
        if self.custom_metric is not None:
            custom_metric = []
            
        # Initialize predictions dictionary if needed
        predictions = {} if self.predictions else None
        
        # Create the preprocessor
        preprocessor = create_preprocessor(X_train)
        
        # Get classifiers to use
        classifiers = self._get_classifiers()
        
        # Fit each classifier
        for name, classifier in tqdm(classifiers, desc="Fitting classifiers"):
            start_time = time.time()
            try:
                # Check if GPU model is available
                if self.use_gpu:
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
                pipe.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipe.predict(X_test)
                
                # Calculate metrics
                acc = accuracy_score(y_test, y_pred, normalize=True)
                bal_acc = balanced_accuracy_score(y_test, y_pred)
                f1_val = f1_score(y_test, y_pred, average="weighted")
                
                # Calculate ROC AUC if binary classification
                if len(np.unique(y_train)) == 2:
                    try:
                        roc_val = roc_auc_score(y_test, y_pred)
                    except:
                        roc_val = np.nan
                else:
                    roc_val = np.nan
                
                # Calculate custom metric if provided
                if self.custom_metric is not None:
                    custom_val = self.custom_metric(y_test, y_pred)
                    custom_metric.append(custom_val)
                
                # Store model and metrics
                self.models[name] = pipe
                names.append(name)
                accuracy.append(acc)
                balanced_accuracy.append(bal_acc)
                roc_auc.append(roc_val)
                f1.append(f1_val)
                time_taken.append(time.time() - start_time)
                
                # Store predictions if requested
                if self.predictions:
                    predictions[name] = y_pred
                
                # Log model performance to MLflow if configured
                if mlflow_tracking_uri:
                    metrics_dict = {
                        "accuracy": acc,
                        "balanced_accuracy": bal_acc,
                        "roc_auc": roc_val if not np.isnan(roc_val) else None,
                        "f1_score": f1_val,
                        "time_taken": time.time() - start_time,
                    }
                    if self.custom_metric is not None:
                        metrics_dict[self.custom_metric.__name__] = custom_val
                    
                    log_model_performance(name, metrics_dict)
                
                # Print progress if verbose
                if self.verbose > 0:
                    metrics_dict = {
                        "Model": name,
                        "Accuracy": acc,
                        "Balanced Accuracy": bal_acc,
                        "ROC AUC": roc_val,
                        "F1 Score": f1_val,
                        "Time taken": time.time() - start_time,
                    }
                    if self.custom_metric is not None:
                        metrics_dict[self.custom_metric.__name__] = custom_val
                        
                    print(metrics_dict)
                    
            except Exception as e:
                if not self.ignore_warnings:
                    logger.error(f"Error fitting {name}: {e}")
        
        # Create results DataFrame
        if self.custom_metric is None:
            scores = pd.DataFrame({
                "Model": names,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy,
                "ROC AUC": roc_auc,
                "F1 Score": f1,
                "Time Taken": time_taken,
            })
        else:
            scores = pd.DataFrame({
                "Model": names,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_accuracy,
                "ROC AUC": roc_auc,
                "F1 Score": f1,
                self.custom_metric.__name__: custom_metric,
                "Time Taken": time_taken,
            })
        
        # Sort by accuracy
        scores = scores.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        
        # Log the full results table to MLflow
        if mlflow_tracking_uri:
            from ..utils import log_dataframe, end_run
            log_dataframe(scores, "model_comparison")
            end_run()
        
        return scores, predictions
        
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
        X_train, _, y_train, _ = self._check_data(X_train, X_train, y_train)
        
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
            if estimator_name == "RandomForestClassifier":
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