"""
Supervised Models - Enhanced LazyPredict with MLflow, Cross-Validation, and Advanced Features
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import numpy as np
import pandas as pd
import sys
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
import json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score,
    precision_score, recall_score, log_loss, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
import warnings
import xgboost
import lightgbm
from scipy import stats

# Import MLflow for model tracking
try:
    import mlflow
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Check if MLflow tracking URI is set
def is_mlflow_tracking_enabled():
    """Checks if MLflow tracking is enabled via environment variable."""
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    return MLFLOW_AVAILABLE and tracking_uri is not None

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 4)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# Enhanced list of removed classifiers/regressors
removed_classifiers = [
    "ClassifierChain", "ComplementNB", "GradientBoostingClassifier",
    "GaussianProcessClassifier", "HistGradientBoostingClassifier", "MLPClassifier",
    "LogisticRegressionCV", "MultiOutputClassifier", "MultinomialNB", 
    "OneVsOneClassifier", "OneVsRestClassifier", "OutputCodeClassifier",
    "RadiusNeighborsClassifier", "VotingClassifier", "StackingClassifier",
]

removed_regressors = [
    "TheilSenRegressor", "ARDRegression", "CCA", "IsotonicRegression", 
    "StackingRegressor", "MultiOutputRegressor", "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV", 
    "PLSCanonical", "PLSRegression", "RadiusNeighborsRegressor", 
    "RegressorChain", "VotingRegressor", "DummyRegressor",
]

# Get all estimators with enhanced filtering
CLASSIFIERS = [
    est for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

REGRESSORS = [
    est for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

# Add popular boosting libraries
REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))

# Enhanced preprocessing pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

robust_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ]
)

power_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", PowerTransformer(method='yeo-johnson'))
    ]
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
        ("encoding", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ]
)

# Helper functions
def get_card_split(df, cols, n=11):
    """Split categorical columns based on cardinality."""
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high

def adjusted_rsquared(r2, n, p):
    """Calculate adjusted R-squared."""
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1)) if n > p + 1 else r2

def calculate_feature_importance(model, feature_names):
    """Calculate feature importance for models that support it."""
    try:
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                return dict(zip(feature_names, np.abs(model.coef_)))
            else:
                return dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
    except:
        return None
    return None

class EnhancedMLflowManager:
    """Manager for MLflow operations."""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.active_runs = {}
        
    def start_run(self, run_name, experiment_name=None):
        """Start an MLflow run."""
        if not self.enabled or not MLFLOW_AVAILABLE:
            return None
            
        try:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=run_name)
            self.active_runs[run_name] = run
            return run
        except Exception as e:
            print(f"MLflow run start failed: {e}")
            return None
            
    def end_run(self, run_name):
        """End an MLflow run."""
        if run_name in self.active_runs:
            try:
                mlflow.end_run()
                del self.active_runs[run_name]
            except Exception as e:
                print(f"MLflow run end failed: {e}")
                
    def log_params(self, params):
        """Log parameters to MLflow."""
        if self.enabled and MLFLOW_AVAILABLE:
            try:
                mlflow.log_params(params)
            except Exception as e:
                print(f"MLflow params logging failed: {e}")
                
    def log_metrics(self, metrics):
        """Log metrics to MLflow."""
        if self.enabled and MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                print(f"MLflow metrics logging failed: {e}")
                
    def log_model(self, model, name, signature=None):
        """Log model to MLflow."""
        if self.enabled and MLFLOW_AVAILABLE:
            try:
                mlflow.sklearn.log_model(model, name, signature=signature)
            except Exception as e:
                print(f"MLflow model logging failed: {e}")

# Main Classifier Class
class LazyClassifier:
    """
    Enhanced LazyClassifier with MLflow, cross-validation, and advanced features.
    """
    
    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
        cv_folds=None,
        feature_selection=False,
        preprocess_strategy='standard',
        mlflow_experiment=None
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers
        self.cv_folds = cv_folds
        self.feature_selection = feature_selection
        self.preprocess_strategy = preprocess_strategy
        self.feature_importances = {}
        
        # MLflow setup
        self.mlflow_manager = EnhancedMLflowManager(enabled=is_mlflow_tracking_enabled())
        self.mlflow_experiment = mlflow_experiment
        
        # Default metrics
        self.metrics = {
            'accuracy': accuracy_score,
            'balanced_accuracy': balanced_accuracy_score,
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score
        }
        
    def _get_preprocessor(self, X_train, categorical_features):
        """Get appropriate preprocessor based on strategy."""
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_low, categorical_high = get_card_split(X_train, categorical_features)
        
        if self.preprocess_strategy == 'robust':
            numeric_transformer_used = robust_numeric_transformer
        elif self.preprocess_strategy == 'power':
            numeric_transformer_used = power_numeric_transformer
        else:
            numeric_transformer_used = numeric_transformer
            
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer_used, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        
        return preprocessor
    
    def _calculate_cv_scores(self, model, X, y, scoring='accuracy'):
        """Calculate cross-validation scores."""
        if self.cv_folds:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        return None
    
    def fit(self, X_train, X_test, y_train, y_test):
        """Fit classification models with enhanced features."""
        results = {
            'Model': [], 'Accuracy': [], 'Balanced_Accuracy': [], 
            'ROC_AUC': [], 'F1_Score': [], 'Time_Taken': [],
            'CV_Accuracy_Mean': [], 'CV_Accuracy_Std': []
        }
        
        if self.custom_metric:
            results[self.custom_metric.__name__] = []
            
        predictions = {}
        
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            
        categorical_features = X_train.select_dtypes(include=["object", "category"]).columns
        preprocessor = self._get_preprocessor(X_train, categorical_features)
        
        # Fit preprocessor to get feature names
        X_train_processed = preprocessor.fit_transform(X_train)
        feature_names = self._get_feature_names(preprocessor)
        
        if self.classifiers == "all":
            classifiers_to_use = CLASSIFIERS
        else:
            classifiers_to_use = [(cls.__name__, cls) for cls in self.classifiers]
        
        progress_bar = notebook_tqdm if use_notebook_tqdm else tqdm
        for name, model_class in progress_bar(classifiers_to_use):
            start_time = time.time()
            mlflow_run = None
            
            try:
                # MLflow run
                if self.mlflow_manager.enabled:
                    mlflow_run = self.mlflow_manager.start_run(
                        f"LazyClassifier-{name}", self.mlflow_experiment
                    )
                    self.mlflow_manager.log_params({
                        'model_name': name,
                        'random_state': self.random_state,
                        'preprocess_strategy': self.preprocess_strategy
                    })
                
                # Model setup
                model_params = {}
                if "random_state" in model_class().get_params().keys():
                    model_params['random_state'] = self.random_state
                if "n_jobs" in model_class().get_params().keys():
                    model_params['n_jobs'] = -1
                    
                model = model_class(**model_params)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                # Training
                pipeline.fit(X_train, y_train)
                self.models[name] = pipeline
                
                # Prediction
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1] if y_pred_proba is not None else y_pred)
                except:
                    metrics['roc_auc'] = None
                
                # Cross-validation
                cv_scores = self._calculate_cv_scores(pipeline, X_train, y_train)
                
                # Feature importance
                feature_importance = calculate_feature_importance(model, feature_names)
                if feature_importance:
                    self.feature_importances[name] = feature_importance
                
                # Store results
                results['Model'].append(name)
                results['Accuracy'].append(metrics['accuracy'])
                results['Balanced_Accuracy'].append(metrics['balanced_accuracy'])
                results['ROC_AUC'].append(metrics['roc_auc'])
                results['F1_Score'].append(metrics['f1'])
                results['Time_Taken'].append(time.time() - start_time)
                
                if cv_scores:
                    results['CV_Accuracy_Mean'].append(cv_scores['mean'])
                    results['CV_Accuracy_Std'].append(cv_scores['std'])
                
                if self.custom_metric:
                    custom_metric_val = self.custom_metric(y_test, y_pred)
                    results[self.custom_metric.__name__].append(custom_metric_val)
                
                # MLflow logging
                if mlflow_run:
                    mlflow_metrics = {k: v for k, v in metrics.items() if v is not None}
                    mlflow_metrics['training_time'] = time.time() - start_time
                    
                    if cv_scores:
                        mlflow_metrics.update({f'cv_{k}': v for k, v in cv_scores.items()})
                    
                    self.mlflow_manager.log_metrics(mlflow_metrics)
                    
                    try:
                        signature = infer_signature(X_train, pipeline.predict(X_train))
                        self.mlflow_manager.log_model(pipeline, f"{name}_model", signature)
                    except Exception as e:
                        if not self.ignore_warnings:
                            print(f"MLflow model logging failed for {name}: {e}")
                
                if self.verbose > 0:
                    self._print_verbose_results(name, metrics, time.time() - start_time, cv_scores)
                
                if self.predictions:
                    predictions[name] = y_pred
                    
            except Exception as e:
                if not self.ignore_warnings:
                    print(f"{name} failed: {e}")
            finally:
                if mlflow_run:
                    self.mlflow_manager.end_run(f"LazyClassifier-{name}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        if 'CV_Accuracy_Mean' in results_df.columns:
            results_df = results_df.sort_values('CV_Accuracy_Mean', ascending=False)
        else:
            results_df = results_df.sort_values('Accuracy', ascending=False)
        
        results_df = results_df.set_index('Model')
        
        if self.predictions:
            return results_df, pd.DataFrame(predictions)
        return results_df
    
    def _get_feature_names(self, preprocessor):
        """Get feature names after preprocessing."""
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'numeric':
                feature_names.extend(features.tolist())
            elif name == 'categorical_low':
                # For one-hot encoding
                ohe = transformer.named_steps['encoding']
                feature_names.extend(ohe.get_feature_names_out(features).tolist())
            elif name == 'categorical_high':
                feature_names.extend(features.tolist())
        return feature_names
    
    def _print_verbose_results(self, name, metrics, time_taken, cv_scores=None):
        """Print verbose results."""
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        if cv_scores:
            print(f"  CV Accuracy: {cv_scores['mean']:.4f} ± {cv_scores['std']:.4f}")
        print(f"  Time: {time_taken:.2f}s")
    
    def provide_models(self, X_train, X_test, y_train, y_test):
        """Return trained models."""
        if not self.models:
            self.fit(X_train, X_test, y_train, y_test)
        return self.models
    
    def get_feature_importances(self, top_n=10):
        """Get feature importances for all models."""
        return self.feature_importances
    
    def get_best_model(self, metric='accuracy'):
        """Get the best model based on specified metric."""
        if not self.models:
            raise ValueError("No models trained yet. Call fit() first.")
        
        # This would need the results from fit(), so you might want to store them
        pass

# Main Regressor Class
class LazyRegressor:
    """
    Enhanced LazyRegressor with MLflow, cross-validation, and advanced features.
    """
    
    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
        cv_folds=None,
        feature_selection=False,
        preprocess_strategy='standard',
        mlflow_experiment=None
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors
        self.cv_folds = cv_folds
        self.feature_selection = feature_selection
        self.preprocess_strategy = preprocess_strategy
        self.feature_importances = {}
        
        # MLflow setup
        self.mlflow_manager = EnhancedMLflowManager(enabled=is_mlflow_tracking_enabled())
        self.mlflow_experiment = mlflow_experiment
        
        # Default metrics
        self.metrics = {
            'r2': r2_score,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error
        }
    
    def fit(self, X_train, X_test, y_train, y_test):
        """Fit regression models with enhanced features."""
        results = {
            'Model': [], 'R2': [], 'Adjusted_R2': [], 
            'RMSE': [], 'MAE': [], 'Time_Taken': [],
            'CV_R2_Mean': [], 'CV_R2_Std': []
        }
        
        if self.custom_metric:
            results[self.custom_metric.__name__] = []
            
        predictions = {}
        
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            
        categorical_features = X_train.select_dtypes(include=["object", "category"]).columns
        preprocessor = self._get_preprocessor(X_train, categorical_features)
        
        # Fit preprocessor to get feature names
        X_train_processed = preprocessor.fit_transform(X_train)
        feature_names = self._get_feature_names(preprocessor)
        
        if self.regressors == "all":
            regressors_to_use = REGRESSORS
        else:
            regressors_to_use = [(reg.__name__, reg) for reg in self.regressors]
        
        progress_bar = notebook_tqdm if use_notebook_tqdm else tqdm
        for name, model_class in progress_bar(regressors_to_use):
            start_time = time.time()
            mlflow_run = None
            
            try:
                # MLflow run
                if self.mlflow_manager.enabled:
                    mlflow_run = self.mlflow_manager.start_run(
                        f"LazyRegressor-{name}", self.mlflow_experiment
                    )
                    self.mlflow_manager.log_params({
                        'model_name': name,
                        'random_state': self.random_state,
                        'preprocess_strategy': self.preprocess_strategy
                    })
                
                # Model setup
                model_params = {}
                if "random_state" in model_class().get_params().keys():
                    model_params['random_state'] = self.random_state
                if "n_jobs" in model_class().get_params().keys():
                    model_params['n_jobs'] = -1
                    
                model = model_class(**model_params)
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                # Training
                pipeline.fit(X_train, y_train)
                self.models[name] = pipeline
                
                # Prediction
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                adj_r2 = adjusted_rsquared(r2, len(y_test), X_test.shape[1])
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                metrics = {
                    'r2': r2,
                    'adj_r2': adj_r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                # Cross-validation
                cv_scores = self._calculate_cv_scores(pipeline, X_train, y_train, 'r2')
                
                # Feature importance
                feature_importance = calculate_feature_importance(model, feature_names)
                if feature_importance:
                    self.feature_importances[name] = feature_importance
                
                # Store results
                results['Model'].append(name)
                results['R2'].append(metrics['r2'])
                results['Adjusted_R2'].append(metrics['adj_r2'])
                results['RMSE'].append(metrics['rmse'])
                results['MAE'].append(metrics['mae'])
                results['Time_Taken'].append(time.time() - start_time)
                
                if cv_scores:
                    results['CV_R2_Mean'].append(cv_scores['mean'])
                    results['CV_R2_Std'].append(cv_scores['std'])
                
                if self.custom_metric:
                    custom_metric_val = self.custom_metric(y_test, y_pred)
                    results[self.custom_metric.__name__].append(custom_metric_val)
                
                # MLflow logging
                if mlflow_run:
                    mlflow_metrics = metrics.copy()
                    mlflow_metrics['training_time'] = time.time() - start_time
                    
                    if cv_scores:
                        mlflow_metrics.update({f'cv_{k}': v for k, v in cv_scores.items()})
                    
                    self.mlflow_manager.log_metrics(mlflow_metrics)
                    
                    try:
                        signature = infer_signature(X_train, pipeline.predict(X_train))
                        self.mlflow_manager.log_model(pipeline, f"{name}_model", signature)
                    except Exception as e:
                        if not self.ignore_warnings:
                            print(f"MLflow model logging failed for {name}: {e}")
                
                if self.verbose > 0:
                    self._print_verbose_results(name, metrics, time.time() - start_time, cv_scores)
                
                if self.predictions:
                    predictions[name] = y_pred
                    
            except Exception as e:
                if not self.ignore_warnings:
                    print(f"{name} failed: {e}")
            finally:
                if mlflow_run:
                    self.mlflow_manager.end_run(f"LazyRegressor-{name}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        if 'CV_R2_Mean' in results_df.columns:
            results_df = results_df.sort_values('CV_R2_Mean', ascending=False)
        else:
            results_df = results_df.sort_values('R2', ascending=False)
        
        results_df = results_df.set_index('Model')
        
        if self.predictions:
            return results_df, pd.DataFrame(predictions)
        return results_df
    
    def _get_preprocessor(self, X_train, categorical_features):
        """Get appropriate preprocessor based on strategy."""
        return LazyClassifier._get_preprocessor(self, X_train, categorical_features)
    
    def _calculate_cv_scores(self, model, X, y, scoring='r2'):
        """Calculate cross-validation scores."""
        if self.cv_folds:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        return None
    
    def _get_feature_names(self, preprocessor):
        """Get feature names after preprocessing."""
        return LazyClassifier._get_feature_names(self, preprocessor)
    
    def _print_verbose_results(self, name, metrics, time_taken, cv_scores=None):
        """Print verbose results."""
        print(f"\n{name}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Adjusted R²: {metrics['adj_r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        if cv_scores:
            print(f"  CV R²: {cv_scores['mean']:.4f} ± {cv_scores['std']:.4f}")
        print(f"  Time: {time_taken:.2f}s")
    
    def provide_models(self, X_train, X_test, y_train, y_test):
        """Return trained models."""
        if not self.models:
            self.fit(X_train, X_test, y_train, y_test)
        return self.models
    
    def get_feature_importances(self, top_n=10):
        """Get feature importances for all models."""
        return self.feature_importances

# Aliases for backward compatibility
Regression = LazyRegressor
Classification = LazyClassifier

# Utility functions
def compare_models(X, y, problem_type='classification', test_size=0.2, **kwargs):
    """Convenience function to quickly compare models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if problem_type == 'classification' else None
    )
    
    if problem_type == 'classification':
        clf = LazyClassifier(**kwargs)
        return clf.fit(X_train, X_test, y_train, y_test)
    else:
        reg = LazyRegressor(**kwargs)
        return reg.fit(X_train, X_test, y_train, y_test)

def get_available_models(problem_type='classification'):
    """Get list of available models."""
    if problem_type == 'classification':
        return [name for name, _ in CLASSIFIERS]
    else:
        return [name for name, _ in REGRESSORS]

if __name__ == "__main__":
    # Example usage
    print("Available classifiers:", get_available_models('classification'))
    print("Available regressors:", get_available_models('regression'))
