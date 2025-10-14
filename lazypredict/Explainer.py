"""
Model Explainability Module for LazyPredict
Provides SHAP-based explanations for trained models with minimal code changes.
"""
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, List, Any
import warnings

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger('lazypredict.explainer')


class ModelExplainer:
    """
    Provides explainability for trained models using SHAP (SHapley Additive exPlanations).

    This class works with any LazyClassifier or LazyRegressor instance that has
    trained models. It generates feature importance and individual prediction
    explanations using SHAP values.

    Parameters
    ----------
    lazy_estimator : LazyClassifier or LazyRegressor
        A fitted LazyClassifier or LazyRegressor instance
    X_train : array-like
        Training features used to fit the models
    X_test : array-like
        Test features for generating explanations

    Attributes
    ----------
    trained_models : dict
        Dictionary of trained model pipelines
    explainers : dict
        Dictionary of SHAP explainer objects for each model
    shap_values : dict
        Dictionary of computed SHAP values for each model

    Notes
    -----
    **Memory Usage Considerations:**
    - SHAP explainers and values are cached for performance
    - For 40+ models, cache can consume 100s of MB
    - Use clear_cache() or clear_model_cache() to free memory when needed
    - Consider explaining only top-performing models to reduce memory footprint

    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from lazypredict.Explainer import ModelExplainer
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> data = load_breast_cancer()
    >>> X, y = data.data, data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>>
    >>> clf = LazyClassifier(verbose=0, ignore_warnings=True)
    >>> models = clf.fit(X_train, X_test, y_train, y_test)
    >>>
    >>> # Create explainer
    >>> explainer = ModelExplainer(clf, X_train, X_test)
    >>>
    >>> # Get feature importance for best model
    >>> importance = explainer.feature_importance('LogisticRegression')
    >>> print(importance.head())
    >>>
    >>> # Plot SHAP summary for a model
    >>> explainer.plot_summary('LogisticRegression')
    >>>
    >>> # Explain a single prediction
    >>> explainer.explain_prediction('LogisticRegression', instance_idx=0)
    >>>
    >>> # Clear cache to free memory
    >>> explainer.clear_cache()
    """
    
    def __init__(
        self,
        lazy_estimator,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
    ):
        """Initialize the explainer with a fitted LazyPredict estimator."""
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required for model explainability. "
                "Install it with: pip install shap"
            )
        
        if not hasattr(lazy_estimator, 'trained_models'):
            raise ValueError(
                "The provided estimator must be a fitted LazyClassifier or LazyRegressor"
            )
        
        if len(lazy_estimator.trained_models) == 0:
            raise ValueError(
                "No trained models found. Please fit the estimator first."
            )
        
        self.lazy_estimator = lazy_estimator
        self.trained_models = lazy_estimator.trained_models
        
        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            self.X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
            self.X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        else:
            self.X_train = X_train.copy()
            self.X_test = X_test.copy()
        
        # Store feature names
        self.feature_names = list(self.X_train.columns)
        
        # Initialize storage
        self.explainers: Dict[str, Any] = {}
        self.shap_values: Dict[str, np.ndarray] = {}
        self.expected_values: Dict[str, float] = {}
        
        logger.info(f"ModelExplainer initialized with {len(self.trained_models)} models")
    
    def _get_explainer(self, model_name: str, max_samples: int = 100) -> Any:
        """
        Create or retrieve a SHAP explainer for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to explain
        max_samples : int
            Maximum number of background samples for TreeExplainer
        
        Returns
        -------
        explainer : shap.Explainer
            SHAP explainer object
        """
        if model_name not in self.trained_models:
            available = list(self.trained_models.keys())
            similar = [m for m in available if model_name.lower() in m.lower()]
            msg = f"Model '{model_name}' not found in trained models.\nAvailable models: {', '.join(available[:5])}"
            if len(available) > 5:
                msg += f" (and {len(available)-5} more)"
            if similar:
                msg += f"\n\nDid you mean one of these?\n  - " + "\n  - ".join(similar[:3])
            raise ValueError(msg)
        
        if model_name in self.explainers:
            return self.explainers[model_name]
        
        pipeline = self.trained_models[model_name]
        model = pipeline.named_steps['model']
        
        # Transform training data using preprocessor
        X_train_transformed = pipeline.named_steps['preprocessor'].transform(self.X_train)
        
        # Choose appropriate explainer based on model type
        try:
            # Tree-based models (XGBoost, LightGBM, RandomForest, etc.)
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_') or \
               type(model).__name__ in ['XGBClassifier', 'XGBRegressor', 
                                        'LGBMClassifier', 'LGBMRegressor',
                                        'RandomForestClassifier', 'RandomForestRegressor',
                                        'ExtraTreesClassifier', 'ExtraTreesRegressor',
                                        'GradientBoostingClassifier', 'GradientBoostingRegressor',
                                        'DecisionTreeClassifier', 'DecisionTreeRegressor']:
                explainer = shap.TreeExplainer(model)
                logger.debug(f"Using TreeExplainer for {model_name}")
            
            # Linear models
            elif type(model).__name__ in ['LogisticRegression', 'LinearRegression', 
                                          'Ridge', 'Lasso', 'ElasticNet',
                                          'SGDClassifier', 'SGDRegressor',
                                          'LinearSVC', 'LinearSVR']:
                explainer = shap.LinearExplainer(model, X_train_transformed)
                logger.debug(f"Using LinearExplainer for {model_name}")
            
            # Kernel explainer as fallback (slower but works with any model)
            else:
                # Use a sample for kernel explainer to reduce computation
                background = shap.sample(X_train_transformed, min(max_samples, len(X_train_transformed)))
                explainer = shap.KernelExplainer(model.predict, background)
                logger.debug(f"Using KernelExplainer for {model_name}")
            
            self.explainers[model_name] = explainer
            return explainer
            
        except Exception as e:
            logger.warning(f"Failed to create explainer for {model_name}: {e}")
            # Fallback to KernelExplainer
            background = shap.sample(X_train_transformed, min(max_samples, len(X_train_transformed)))
            explainer = shap.KernelExplainer(model.predict, background)
            self.explainers[model_name] = explainer
            return explainer
    
    def _compute_shap_values(self, model_name: str, max_samples: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for a model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        max_samples : int, optional
            Maximum number of test samples to compute SHAP values for
        
        Returns
        -------
        shap_values : np.ndarray
            SHAP values for the test set
        """
        if model_name in self.shap_values:
            return self.shap_values[model_name]
        
        explainer = self._get_explainer(model_name)
        pipeline = self.trained_models[model_name]
        
        # Transform test data
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(self.X_test)
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(X_test_transformed):
            X_test_transformed = X_test_transformed[:max_samples]
        
        # Compute SHAP values
        try:
            shap_values = explainer.shap_values(X_test_transformed)
            
            # Handle multi-class classification (take values for positive class)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Positive class for binary
            
            self.shap_values[model_name] = shap_values
            
            # Store expected value
            if hasattr(explainer, 'expected_value'):
                expected = explainer.expected_value
                if isinstance(expected, (list, np.ndarray)):
                    expected = expected[1] if len(expected) > 1 else expected[0]
                self.expected_values[model_name] = expected
            
            logger.info(f"Computed SHAP values for {model_name}")
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to compute SHAP values for {model_name}: {e}")
            raise
    
    def feature_importance(
        self,
        model_name: str,
        top_n: Optional[int] = None,
        absolute: bool = True
    ) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.

        Parameters
        ----------
        model_name : str
            Name of the model to explain
        top_n : int, optional
            Return only top N features (must be positive if provided)
        absolute : bool, default=True
            Use absolute SHAP values (measures impact regardless of direction)

        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with features and their importance scores

        Examples
        --------
        >>> importance = explainer.feature_importance('LogisticRegression', top_n=10)
        >>> print(importance)
        """
        if top_n is not None and top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")

        shap_values = self._compute_shap_values(model_name)
        
        # Calculate mean absolute SHAP value for each feature
        if absolute:
            importance_scores = np.abs(shap_values).mean(axis=0)
        else:
            importance_scores = shap_values.mean(axis=0)
        
        # For multi-class classification, SHAP returns shape (n_samples, n_features, n_classes)
        # After mean(axis=0), we get (n_features, n_classes)
        # We need to average across classes to get (n_features,)
        if importance_scores.ndim > 1:
            importance_scores = importance_scores.mean(axis=1)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df.reset_index(drop=True)
    
    def plot_summary(
        self,
        model_name: str,
        plot_type: str = 'dot',
        max_display: int = 20,
        show: bool = True
    ) -> None:
        """
        Create a SHAP summary plot showing feature importance and effects.
        
        Parameters
        ----------
        model_name : str
            Name of the model to explain
        plot_type : str, default='dot'
            Type of plot ('dot', 'bar', 'violin')
        max_display : int, default=20
            Maximum number of features to display
        show : bool, default=True
            Whether to display the plot immediately
        
        Examples
        --------
        >>> explainer.plot_summary('RandomForestClassifier', plot_type='bar')
        """
        shap_values = self._compute_shap_values(model_name)
        pipeline = self.trained_models[model_name]
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(self.X_test)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_values, 
                X_test_transformed, 
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_values, 
                X_test_transformed, 
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )
        
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        
        if show:
            plt.show()
    
    def explain_prediction(
        self,
        model_name: str,
        instance_idx: int = 0,
        show: bool = True
    ) -> None:
        """
        Explain a single prediction using a waterfall plot.
        
        Parameters
        ----------
        model_name : str
            Name of the model to explain
        instance_idx : int, default=0
            Index of the test instance to explain
        show : bool, default=True
            Whether to display the plot immediately
        
        Examples
        --------
        >>> explainer.explain_prediction('XGBClassifier', instance_idx=5)
        """
        shap_values = self._compute_shap_values(model_name)
        
        if instance_idx >= len(shap_values):
            raise ValueError(f"instance_idx {instance_idx} is out of range (max: {len(shap_values)-1})")
        
        # Get the SHAP values for this instance
        instance_shap = shap_values[instance_idx]
        
        # Get feature values for this instance
        pipeline = self.trained_models[model_name]
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(self.X_test)
        instance_features = X_test_transformed[instance_idx]
        
        # Create explanation object
        expected_value = self.expected_values.get(model_name, 0)
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        
        try:
            # For newer SHAP versions
            explanation = shap.Explanation(
                values=instance_shap,
                base_values=expected_value,
                data=instance_features,
                feature_names=self.feature_names
            )
            shap.waterfall_plot(explanation, show=False)
        except:
            # Fallback to force plot for older versions
            shap.force_plot(
                expected_value,
                instance_shap,
                instance_features,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
        
        plt.title(f'Prediction Explanation - {model_name} (Instance {instance_idx})')
        plt.tight_layout()
        
        if show:
            plt.show()
    
    def plot_dependence(
        self,
        model_name: str,
        feature: Union[str, int],
        interaction_feature: Optional[Union[str, int]] = 'auto',
        show: bool = True
    ) -> None:
        """
        Create a dependence plot showing how a feature affects predictions.
        
        Parameters
        ----------
        model_name : str
            Name of the model to explain
        feature : str or int
            Feature name or index to plot
        interaction_feature : str, int, or 'auto', optional
            Feature to color by for interaction effects
        show : bool, default=True
            Whether to display the plot immediately
        
        Examples
        --------
        >>> explainer.plot_dependence('LogisticRegression', 'age', interaction_feature='auto')
        """
        shap_values = self._compute_shap_values(model_name)
        pipeline = self.trained_models[model_name]
        X_test_transformed = pipeline.named_steps['preprocessor'].transform(self.X_test)
        
        # Convert feature name to index if needed
        if isinstance(feature, str):
            if feature not in self.feature_names:
                raise ValueError(f"Feature '{feature}' not found")
            feature_idx = self.feature_names.index(feature)
        else:
            feature_idx = feature
            feature = self.feature_names[feature_idx]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X_test_transformed,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.title(f'SHAP Dependence Plot - {model_name} - {feature}')
        plt.tight_layout()
        
        if show:
            plt.show()
    
    def compare_models(
        self,
        model_names: Optional[List[str]] = None,
        top_n_features: int = 10,
        show: bool = True
    ) -> pd.DataFrame:
        """
        Compare feature importance across multiple models.
        
        Parameters
        ----------
        model_names : list of str, optional
            List of model names to compare. If None, uses all trained models.
        top_n_features : int, default=10
            Number of top features to include in comparison
        show : bool, default=True
            Whether to display a comparison plot
        
        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame comparing feature importance across models
        
        Examples
        --------
        >>> comparison = explainer.compare_models(['LogisticRegression', 'RandomForest'])
        >>> print(comparison)
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())[:5]  # Limit to 5 for clarity
        
        # Collect feature importance from each model
        importance_dict = {}
        for model_name in model_names:
            try:
                importance = self.feature_importance(model_name, absolute=True)
                importance_dict[model_name] = importance.set_index('feature')['importance']
            except Exception as e:
                logger.warning(f"Failed to compute importance for {model_name}: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(importance_dict)
        
        # Get top features across all models
        comparison_df['mean_importance'] = comparison_df.mean(axis=1)
        comparison_df = comparison_df.sort_values('mean_importance', ascending=False)
        comparison_df = comparison_df.head(top_n_features)
        comparison_df = comparison_df.drop('mean_importance', axis=1)
        
        # Create visualization
        if show and len(comparison_df) > 0:
            plt.figure(figsize=(12, 6))
            comparison_df.plot(kind='bar', figsize=(12, 6))
            plt.title('Feature Importance Comparison Across Models')
            plt.xlabel('Feature')
            plt.ylabel('Mean Absolute SHAP Value')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
        
        return comparison_df
    
    def get_top_features(
        self,
        model_name: str,
        instance_idx: int,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Get the top features contributing to a specific prediction.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        instance_idx : int
            Index of the test instance
        top_n : int, default=5
            Number of top features to return
        
        Returns
        -------
        top_features_df : pd.DataFrame
            DataFrame with top features and their SHAP values
        
        Examples
        --------
        >>> top_features = explainer.get_top_features('XGBClassifier', instance_idx=0, top_n=5)
        >>> print(top_features)
        """
        shap_values = self._compute_shap_values(model_name)
        
        if instance_idx >= len(shap_values):
            raise ValueError(f"instance_idx {instance_idx} is out of range")
        
        instance_shap = shap_values[instance_idx]
        
        # Get absolute SHAP values
        abs_shap = np.abs(instance_shap)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        top_features_df = pd.DataFrame({
            'feature': [self.feature_names[i] for i in top_indices],
            'shap_value': [instance_shap[i] for i in top_indices],
            'abs_shap_value': [abs_shap[i] for i in top_indices]
        })

        return top_features_df

    def clear_cache(self) -> None:
        """
        Clear all cached SHAP explainers and values to free memory.

        This is useful when working with many models and memory usage becomes
        a concern. After clearing the cache, SHAP values will be recomputed
        on the next method call.

        Examples
        --------
        >>> explainer = ModelExplainer(clf, X_train, X_test)
        >>> # Analyze several models...
        >>> explainer.clear_cache()  # Free memory
        >>> # Continue with other models...
        """
        self.explainers.clear()
        self.shap_values.clear()
        self.expected_values.clear()
        logger.info("Cleared all cached SHAP explainers and values")

    def clear_model_cache(self, model_name: str) -> None:
        """
        Clear cached SHAP explainer and values for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model whose cache should be cleared

        Examples
        --------
        >>> explainer.clear_model_cache('RandomForestClassifier')
        """
        if model_name in self.explainers:
            del self.explainers[model_name]
        if model_name in self.shap_values:
            del self.shap_values[model_name]
        if model_name in self.expected_values:
            del self.expected_values[model_name]
        logger.info(f"Cleared cache for model: {model_name}")
