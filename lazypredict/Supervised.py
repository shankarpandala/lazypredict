"""
Supervised Models with Improvements
- Type hints
- Logging framework
- Base class to reduce duplication
- Parallel training support
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

# Fix matplotlib backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')

import logging
import numpy as np
import pandas as pd
import sys
from typing import Union, Optional, Tuple, Dict, List, Callable, Any
from tqdm import tqdm
try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        from tqdm.notebook import tqdm as notebook_tqdm
        use_notebook_tqdm = True
    else:
        use_notebook_tqdm = False
except:
    use_notebook_tqdm = False

import datetime
import time
import os
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
import warnings
import xgboost
import lightgbm
from joblib import Parallel, delayed

# Import MLflow for model tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lazypredict')

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelList = List[Tuple[str, type]]

# Model lists
removed_classifiers = [
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

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
]

CLASSIFIERS: ModelList = [
    est for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

REGRESSORS: ModelList = [
    est for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))

# Preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder()),
    ]
)


def get_card_split(df: pd.DataFrame, cols: pd.Index, n: int = 11) -> Tuple[pd.Index, pd.Index]:
    """
    Splits categorical columns into 2 lists based on cardinality.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : pd.Index
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    
    Returns
    -------
    card_low : pd.Index
        Columns with cardinality < n
    card_high : pd.Index
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def is_mlflow_tracking_enabled() -> bool:
    """Checks if MLflow tracking is enabled via environment variable."""
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    return MLFLOW_AVAILABLE and tracking_uri is not None


def setup_mlflow() -> bool:
    """Initialize MLflow if tracking URI is set through environment variable."""
    if is_mlflow_tracking_enabled():
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.autolog()
        logger.info(f"MLflow tracking enabled with URI: {tracking_uri}")
        return True
    return False


def adjusted_rsquared(r2: float, n: int, p: int) -> float:
    """Calculate adjusted R-squared."""
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


class BaseLazyEstimator(ABC):
    """
    Base class for LazyClassifier and LazyRegressor.
    Extracts common logic to reduce code duplication.
    """
    
    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        models: Union[str, List[type]] = "all",
        n_jobs: int = 1,
    ):
        """
        Initialize base estimator.
        
        Parameters
        ----------
        verbose : int, optional (default=0)
            Verbosity level. Set to positive number for detailed output.
        ignore_warnings : bool, optional (default=True)
            When True, warnings from models are ignored.
        custom_metric : function, optional (default=None)
            Custom evaluation metric function.
        predictions : bool, optional (default=False)
            When True, return predictions of all models.
        random_state : int, optional (default=42)
            Random state for reproducibility.
        models : str or list, optional (default="all")
            Models to train. "all" or list of model classes.
        n_jobs : int, optional (default=1)
            Number of parallel jobs. -1 uses all processors.
        """
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.random_state = random_state
        self.models_param = models
        self.n_jobs = n_jobs
        self.trained_models: Dict[str, Pipeline] = {}
        self.mlflow_enabled = setup_mlflow()
        
        # Configure logger verbosity
        if verbose > 0:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
    
    def _prepare_data(
        self, 
        X_train: ArrayLike, 
        X_test: ArrayLike
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert arrays to DataFrames if needed."""
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        return X_train, X_test
    
    def _get_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline based on data types."""
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns
        
        categorical_low, categorical_high = get_card_split(X_train, categorical_features)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        return preprocessor
    
    def _prepare_models(self, all_models: ModelList) -> ModelList:
        """Prepare list of models to train."""
        if self.models_param == "all":
            return all_models
        else:
            try:
                temp_list = []
                for model in self.models_param:
                    # Handle both model classes and string names
                    if isinstance(model, str):
                        # Find model by name in all_models
                        found = False
                        for name, model_class in all_models:
                            if name == model:
                                temp_list.append((name, model_class))
                                found = True
                                break
                        if not found:
                            logger.warning(f"Model '{model}' not found in available models")
                    else:
                        # Assume it's a model class
                        full_name = (model.__name__, model)
                        temp_list.append(full_name)
                return temp_list
            except Exception as e:
                logger.error(f"Invalid model(s): {e}")
                raise ValueError(f"Invalid model(s): {e}")
    
    def _train_single_model(
        self,
        name: str,
        model_class: type,
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: ArrayLike,
        X_test: pd.DataFrame,
        y_test: ArrayLike,
    ) -> Optional[Dict[str, Any]]:
        """Train a single model and return metrics."""
        start_time = time.time()
        mlflow_run = None
        
        try:
            # Start MLflow run
            if self.mlflow_enabled and MLFLOW_AVAILABLE:
                mlflow_run = mlflow.start_run(run_name=f"{self.__class__.__name__}-{name}")
                mlflow.log_param("model_name", name)
            
            # Create pipeline
            if "random_state" in model_class().get_params().keys():
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model_class(random_state=self.random_state)),
                ])
            else:
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model_class()),
                ])
            
            # Train model
            pipe.fit(X_train, y_train)
            self.trained_models[name] = pipe
            
            # Make predictions
            y_pred = pipe.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, X_test, name)
            metrics["Time Taken"] = time.time() - start_time
            metrics["Model"] = name
            
            # Log to MLflow
            if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_run:
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ["Model"] and metric_value is not None:
                        mlflow.log_metric(metric_name.lower().replace(" ", "_"), float(metric_value))
                
                try:
                    signature = mlflow.models.infer_signature(X_train, pipe.predict(X_train))
                    mlflow.sklearn.log_model(
                        pipe, f"{name}_model", 
                        signature=signature,
                        registered_model_name=f"{self.__class__.__name__.lower()}_{name}"
                    )
                except Exception as e:
                    if not self.ignore_warnings:
                        logger.warning(f"Failed to log model {name} to MLflow: {e}")
            
            logger.debug(f"Completed {name}: {metrics}")
            
            # End MLflow run
            if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_run:
                mlflow.end_run()
            
            return {
                "metrics": metrics,
                "predictions": y_pred if self.predictions else None
            }
            
        except Exception as e:
            if self.mlflow_enabled and MLFLOW_AVAILABLE and mlflow_run:
                mlflow.end_run()
            
            if not self.ignore_warnings:
                logger.error(f"{name} model failed: {e}")
            return None
    
    @abstractmethod
    def _calculate_metrics(
        self, 
        y_test: ArrayLike, 
        y_pred: ArrayLike,
        X_test: pd.DataFrame,
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate model-specific metrics. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_all_models(self) -> ModelList:
        """Get list of all models. Implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_sort_column(self) -> str:
        """Get column name to sort results. Implemented by subclasses."""
        pass
    
    def fit(
        self, 
        X_train: ArrayLike, 
        X_test: ArrayLike, 
        y_train: ArrayLike, 
        y_test: ArrayLike
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fit models to training data and evaluate on test data.
        
        Parameters
        ----------
        X_train : array-like
            Training features
        X_test : array-like
            Testing features
        y_train : array-like
            Training targets
        y_test : array-like
            Testing targets
        
        Returns
        -------
        scores : pd.DataFrame
            Metrics for all models
        predictions : pd.DataFrame, optional
            Predictions from all models (if predictions=True)
        """
        logger.info(f"Starting {self.__class__.__name__} fit")
        
        # Prepare data
        X_train, X_test = self._prepare_data(X_train, X_test)
        
        # Get preprocessor
        preprocessor = self._get_preprocessor(X_train)
        
        # Prepare models
        models = self._prepare_models(self._get_all_models())
        logger.info(f"Training {len(models)} models")
        
        # Train models (parallel or sequential)
        if self.n_jobs == 1:
            results = self._train_sequential(models, preprocessor, X_train, y_train, X_test, y_test)
        else:
            results = self._train_parallel(models, preprocessor, X_train, y_train, X_test, y_test)
        
        # Collect metrics and predictions
        all_metrics = []
        all_predictions = {}
        
        for result in results:
            if result is not None:
                all_metrics.append(result["metrics"])
                if self.predictions and result["predictions"] is not None:
                    all_predictions[result["metrics"]["Model"]] = result["predictions"]
        
        # Create DataFrame
        scores = pd.DataFrame(all_metrics)
        scores = scores.sort_values(by=self._get_sort_column(), ascending=False).set_index("Model")
        
        logger.info(f"Completed training {len(scores)} models successfully")
        
        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(all_predictions)
            return scores, predictions_df
        return scores
    
    def _train_sequential(
        self,
        models: ModelList,
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: ArrayLike,
        X_test: pd.DataFrame,
        y_test: ArrayLike,
    ) -> List[Optional[Dict]]:
        """Train models sequentially with progress bar."""
        progress_bar = notebook_tqdm if use_notebook_tqdm else tqdm
        results = []
        for name, model_class in progress_bar(models):
            result = self._train_single_model(
                name, model_class, preprocessor, X_train, y_train, X_test, y_test
            )
            results.append(result)
        return results
    
    def _train_parallel(
        self,
        models: ModelList,
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: ArrayLike,
        X_test: pd.DataFrame,
        y_test: ArrayLike,
    ) -> List[Optional[Dict]]:
        """Train models in parallel using joblib."""
        logger.info(f"Training models in parallel with n_jobs={self.n_jobs}")
        
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._train_single_model)(
                name, model_class, preprocessor, X_train, y_train, X_test, y_test
            )
            for name, model_class in models
        )
        return results
    
    def provide_models(
        self, 
        X_train: ArrayLike, 
        X_test: ArrayLike, 
        y_train: ArrayLike, 
        y_test: ArrayLike
    ) -> Dict[str, Pipeline]:
        """
        Get trained model objects.
        
        Parameters
        ----------
        X_train, X_test, y_train, y_test : array-like
            Training and testing data
        
        Returns
        -------
        models : dict
            Dictionary of trained model pipelines
        """
        if len(self.trained_models) == 0:
            self.fit(X_train, X_test, y_train, y_test)
        return self.trained_models


class LazyClassifier(BaseLazyEstimator):
    """
    Automated classification model selection and evaluation.
    
    This class fits multiple classification algorithms and evaluates them
    on test data, returning a sorted DataFrame of metrics.
    
    Parameters
    ----------
    verbose : int, optional (default=0)
        Verbosity level
    ignore_warnings : bool, optional (default=True)
        Ignore model warnings
    custom_metric : function, optional (default=None)
        Custom evaluation metric
    predictions : bool, optional (default=False)
        Return predictions
    random_state : int, optional (default=42)
        Random state for reproducibility
    classifiers : str or list, optional (default="all")
        Classifiers to train
    n_jobs : int, optional (default=1)
        Number of parallel jobs. -1 for all processors.
    
    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> data = load_breast_cancer()
    >>> X, y = data.data, data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> 
    >>> clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=-1)
    >>> models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> print(models.head())
    """
    
    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        classifiers: Union[str, List[type]] = "all",
        n_jobs: int = 1,
    ):
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            models=classifiers,
            n_jobs=n_jobs,
        )
    
    def _get_all_models(self) -> ModelList:
        """Get all available classifiers."""
        return CLASSIFIERS
    
    def _get_sort_column(self) -> str:
        """Sort by balanced accuracy."""
        return "Balanced Accuracy"
    
    def _calculate_metrics(
        self, 
        y_test: ArrayLike, 
        y_pred: ArrayLike,
        X_test: pd.DataFrame,
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        }
        
        # Try to calculate ROC AUC
        try:
            metrics["ROC AUC"] = roc_auc_score(y_test, y_pred)
        except Exception as e:
            metrics["ROC AUC"] = None
            if not self.ignore_warnings:
                logger.warning(f"ROC AUC couldn't be calculated for {model_name}: {e}")
        
        # Calculate custom metric
        if self.custom_metric is not None:
            try:
                custom_value = self.custom_metric(y_test, y_pred)
                metrics[self.custom_metric.__name__] = custom_value
            except Exception as e:
                if not self.ignore_warnings:
                    logger.warning(f"Custom metric failed for {model_name}: {e}")
        
        return metrics


class LazyRegressor(BaseLazyEstimator):
    """
    Automated regression model selection and evaluation.
    
    This class fits multiple regression algorithms and evaluates them
    on test data, returning a sorted DataFrame of metrics.
    
    Parameters
    ----------
    verbose : int, optional (default=0)
        Verbosity level
    ignore_warnings : bool, optional (default=True)
        Ignore model warnings
    custom_metric : function, optional (default=None)
        Custom evaluation metric
    predictions : bool, optional (default=False)
        Return predictions
    random_state : int, optional (default=42)
        Random state for reproducibility
    regressors : str or list, optional (default="all")
        Regressors to train
    n_jobs : int, optional (default=1)
        Number of parallel jobs. -1 for all processors.
    
    Examples
    --------
    >>> from lazypredict.Supervised import LazyRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> data = load_diabetes()
    >>> X, y = data.data, data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> 
    >>> reg = LazyRegressor(verbose=0, ignore_warnings=True, n_jobs=-1)
    >>> models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    >>> print(models.head())
    """
    
    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        custom_metric: Optional[Callable] = None,
        predictions: bool = False,
        random_state: int = 42,
        regressors: Union[str, List[type]] = "all",
        n_jobs: int = 1,
    ):
        super().__init__(
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            random_state=random_state,
            models=regressors,
            n_jobs=n_jobs,
        )
    
    def _get_all_models(self) -> ModelList:
        """Get all available regressors."""
        return REGRESSORS
    
    def _get_sort_column(self) -> str:
        """Sort by adjusted R-squared."""
        return "Adjusted R-Squared"
    
    def _calculate_metrics(
        self, 
        y_test: ArrayLike, 
        y_pred: ArrayLike,
        X_test: pd.DataFrame,
        model_name: str
    ) -> Dict[str, Any]:
        """Calculate regression metrics."""
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_rsquared(r2, X_test.shape[0], X_test.shape[1])
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics = {
            "R-Squared": r2,
            "Adjusted R-Squared": adj_r2,
            "RMSE": rmse,
        }
        
        # Calculate custom metric
        if self.custom_metric is not None:
            try:
                custom_value = self.custom_metric(y_test, y_pred)
                metrics[self.custom_metric.__name__] = custom_value
            except Exception as e:
                if not self.ignore_warnings:
                    logger.warning(f"Custom metric failed for {model_name}: {e}")
        
        return metrics


# Backward compatibility aliases
Regression = LazyRegressor
Classification = LazyClassifier
