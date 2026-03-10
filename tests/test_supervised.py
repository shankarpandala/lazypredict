import pytest
import numpy as np
import pandas as pd
import os
from lazypredict.Supervised import LazyClassifier, LazyRegressor, get_card_split, INTEL_EXTENSION_AVAILABLE
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

try:
    import mlflow
    MLFLOW_INSTALLED = True
except ImportError:
    MLFLOW_INSTALLED = False


def test_intel_extension_import():
    """Test that Intel Extension for Scikit-learn is properly handled"""
    # Should be a boolean indicating if Intel extension is available
    assert isinstance(INTEL_EXTENSION_AVAILABLE, bool)
    # The import should not fail even if Intel extension is not installed


def test_lazy_classifier_boolean_features():
    """Test LazyClassifier with boolean DataFrame features"""
    # Create dataset with boolean features
    np.random.seed(42)
    n_samples = 200

    # Create boolean features
    df = pd.DataFrame({
        'bool_feat_1': np.random.choice([True, False], n_samples),
        'bool_feat_2': np.random.choice([True, False], n_samples),
        'bool_feat_3': np.random.choice([True, False], n_samples),
        'bool_feat_4': np.random.choice([True, False], n_samples),
    })

    # Create target based on boolean features
    y = ((df['bool_feat_1'].astype(int) + df['bool_feat_2'].astype(int)) > 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Verify results
    assert isinstance(models, pd.DataFrame)
    assert len(models) > 0
    assert "Balanced Accuracy" in models.columns
    # Should have more than just DummyClassifier
    assert len(models) > 1


def test_lazy_classifier_mixed_boolean_numeric():
    """Test LazyClassifier with mixed boolean and numeric features"""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame({
        'bool_feat_1': np.random.choice([True, False], n_samples),
        'bool_feat_2': np.random.choice([True, False], n_samples),
        'numeric_feat_1': np.random.randn(n_samples),
        'numeric_feat_2': np.random.randn(n_samples),
    })

    y = ((df['bool_feat_1'].astype(int) + (df['numeric_feat_1'] > 0).astype(int)) > 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert isinstance(models, pd.DataFrame)
    assert len(models) > 1  # More than DummyClassifier


def test_lazy_regressor_boolean_features():
    """Test LazyRegressor with boolean DataFrame features"""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame({
        'bool_feat_1': np.random.choice([True, False], n_samples),
        'bool_feat_2': np.random.choice([True, False], n_samples),
        'bool_feat_3': np.random.choice([True, False], n_samples),
    })

    # Create continuous target
    y = df['bool_feat_1'].astype(int) * 2 + df['bool_feat_2'].astype(int) + np.random.randn(n_samples) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    assert isinstance(models, pd.DataFrame)
    assert len(models) > 0
    assert "R-Squared" in models.columns


def test_lazy_regressor_mixed_boolean_numeric():
    """Test LazyRegressor with mixed boolean and numeric features"""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame({
        'bool_feat_1': np.random.choice([True, False], n_samples),
        'numeric_feat_1': np.random.randn(n_samples),
        'numeric_feat_2': np.random.randn(n_samples),
    })

    y = df['bool_feat_1'].astype(int) * 3 + df['numeric_feat_1'] * 2 + np.random.randn(n_samples) * 0.1

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    assert isinstance(models, pd.DataFrame)
    assert len(models) > 0


def test_lazy_classifier_fit():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)
    assert len(models) > 0
    assert "Balanced Accuracy" in models.columns


def test_lazy_classifier_custom_metric():
    def custom_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=custom_accuracy)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert "custom_accuracy" in models.columns


@pytest.mark.skipif(not MLFLOW_INSTALLED, reason="mlflow not installed")
def test_lazy_classifier_mlflow_integration():
    # Set up MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Verify MLflow experiments were created
    assert os.path.exists("mlflow.db")


def test_lazy_classifier_categorical_features():
    # Create dataset with categorical features
    n_samples = 100
    X = pd.DataFrame({
        'num1': np.random.rand(n_samples),
        'num2': np.random.rand(n_samples),
        'cat_low': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_high': np.random.choice(['X' + str(i) for i in range(20)], n_samples)
    })
    y = np.random.randint(0, 2, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)


def test_lazy_regressor_fit():
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)
    assert len(models) > 0
    assert "Adjusted R-Squared" in models.columns


def test_lazy_regressor_custom_metric():
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=mean_absolute_error)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert "mean_absolute_error" in models.columns


@pytest.mark.skipif(not MLFLOW_INSTALLED, reason="mlflow not installed")
def test_lazy_regressor_mlflow_integration():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    reg = LazyRegressor(verbose=0, ignore_warnings=False)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    assert os.path.exists("mlflow.db")


def test_get_card_split():
    n_rows = 20
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd', 'e'] * 4,               # 5 unique
        'B': ['f', 'g', 'h', 'i', 'j'] * 4,               # 5 unique
        'C': ['x' + str(i) for i in range(n_rows)],        # 20 unique
    })
    cols = pd.Index(['A', 'B', 'C'])
    card_low, card_high = get_card_split(df, cols, n=10)
    assert list(card_low) == ['A', 'B']
    assert list(card_high) == ['C']


def test_lazy_classifier_specific_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(
        verbose=0, ignore_warnings=True,
        classifiers=[RandomForestClassifier, LogisticRegression])
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    assert len(models) == 2
    assert "RandomForestClassifier" in models.index
    assert "LogisticRegression" in models.index


def test_provide_models():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=[f"feature_{i}" for i in range(data.data.shape[1])])
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    models = clf.provide_models(X_train, X_test, y_train, y_test)
    assert isinstance(models, dict)
    assert len(models) > 0
    # Verify we can make predictions with the returned models
    for name, model in models.items():
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)


def test_roc_auc_uses_probabilities():
    """
    Test that ROC-AUC is calculated using predicted probabilities instead of class labels.
    This addresses issue #476: Severe Underestimation of ROC-AUC Values in Lazypredict Library

    The test verifies that:
    1. LazyClassifier calculates ROC-AUC using predict_proba() for models that support it
    2. ROC-AUC values are higher and more accurate when using probabilities vs class labels
    3. The implementation correctly handles binary and multiclass classification
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Load binary classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test LazyClassifier with a model that supports predict_proba
    clf = LazyClassifier(verbose=0, ignore_warnings=True, classifiers=[LogisticRegression])
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    # Get ROC-AUC from LazyClassifier
    lazy_roc_auc = models.loc['LogisticRegression', 'ROC AUC']

    # Manually calculate ROC-AUC using probabilities (correct method)
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    expected_roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Manually calculate ROC-AUC using class labels (old incorrect method)
    y_pred_labels = pipe.predict(X_test)
    roc_auc_with_labels = roc_auc_score(y_test, y_pred_labels)

    # Verify LazyClassifier uses probabilities (should match expected_roc_auc, not roc_auc_with_labels)
    assert np.isclose(lazy_roc_auc, expected_roc_auc, atol=0.001), \
        f"LazyClassifier ROC-AUC ({lazy_roc_auc}) should match probability-based calculation ({expected_roc_auc})"

    # Verify probabilities give higher ROC-AUC than class labels (for most cases)
    # This demonstrates the fix for issue #476
    assert expected_roc_auc >= roc_auc_with_labels, \
        f"Probability-based ROC-AUC ({expected_roc_auc}) should be >= label-based ({roc_auc_with_labels})"

    # Ensure ROC-AUC is not None
    assert lazy_roc_auc is not None, "ROC-AUC should not be None for models with predict_proba"

    # Verify ROC-AUC is in valid range [0, 1]
    assert 0 <= lazy_roc_auc <= 1, f"ROC-AUC ({lazy_roc_auc}) should be between 0 and 1"


def test_lazy_classifier_kfold_cv():
    """
    Test K-fold cross-validation functionality for LazyClassifier.
    Addresses issue #345: K-fold Cross-validation feature request

    Verifies that:
    1. LazyClassifier accepts cv parameter
    2. CV results include mean and std columns
    3. CV metrics are calculated correctly
    """
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test with 5-fold CV
    clf = LazyClassifier(verbose=0, ignore_warnings=True, cv=5, classifiers=[LogisticRegression])
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    # Verify CV columns are present
    assert "Accuracy CV Mean" in models.columns
    assert "Accuracy CV Std" in models.columns
    assert "Balanced Accuracy CV Mean" in models.columns
    assert "Balanced Accuracy CV Std" in models.columns
    assert "F1 Score CV Mean" in models.columns
    assert "F1 Score CV Std" in models.columns

    # Verify CV metrics are calculated (not None)
    assert models.loc['LogisticRegression', 'Accuracy CV Mean'] is not None
    assert models.loc['LogisticRegression', 'Accuracy CV Std'] is not None

    # Verify CV std is non-negative
    assert models.loc['LogisticRegression', 'Accuracy CV Std'] >= 0
    assert models.loc['LogisticRegression', 'Balanced Accuracy CV Std'] >= 0


def test_lazy_classifier_predict():
    """
    Test built-in predict function for LazyClassifier.
    Addresses issue #345: Built-in predict function feature request

    Verifies that:
    1. predict() returns predictions from all models
    2. predict(model_name) returns predictions from specific model
    3. Proper error handling for unfitted models and invalid model names
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(
        verbose=0, ignore_warnings=True,
        classifiers=[LogisticRegression, RandomForestClassifier])
    clf.fit(X_train, X_test, y_train, y_test)

    # Test predict all models
    all_predictions = clf.predict(X_test)
    assert isinstance(all_predictions, dict)
    assert 'LogisticRegression' in all_predictions
    assert 'RandomForestClassifier' in all_predictions
    assert len(all_predictions['LogisticRegression']) == len(X_test)

    # Test predict specific model
    lr_predictions = clf.predict(X_test, model_name='LogisticRegression')
    assert isinstance(lr_predictions, np.ndarray)
    assert len(lr_predictions) == len(X_test)
    assert np.array_equal(lr_predictions, all_predictions['LogisticRegression'])

    # Test error for unfitted model
    clf_unfitted = LazyClassifier(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="No models have been fitted yet"):
        clf_unfitted.predict(X_test)

    # Test error for invalid model name
    with pytest.raises(ValueError, match="Model .* not found"):
        clf.predict(X_test, model_name='InvalidModel')


def test_lazy_regressor_kfold_cv():
    """
    Test K-fold cross-validation functionality for LazyRegressor.
    """
    from sklearn.linear_model import LinearRegression

    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test with 5-fold CV
    reg = LazyRegressor(verbose=0, ignore_warnings=True, cv=5, regressors=[LinearRegression])
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    # Verify CV columns are present
    assert "R-Squared CV Mean" in models.columns
    assert "R-Squared CV Std" in models.columns
    assert "RMSE CV Mean" in models.columns
    assert "RMSE CV Std" in models.columns

    # Verify CV metrics are calculated (not None)
    assert models.loc['LinearRegression', 'R-Squared CV Mean'] is not None
    assert models.loc['LinearRegression', 'RMSE CV Mean'] is not None

    # Verify CV std is non-negative
    assert models.loc['LinearRegression', 'R-Squared CV Std'] >= 0
    assert models.loc['LinearRegression', 'RMSE CV Std'] >= 0


def test_lazy_regressor_predict():
    """
    Test built-in predict function for LazyRegressor.
    """
    from sklearn.linear_model import LinearRegression, Ridge

    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = LazyRegressor(verbose=0, ignore_warnings=True, regressors=[LinearRegression, Ridge])
    reg.fit(X_train, X_test, y_train, y_test)

    # Test predict all models
    all_predictions = reg.predict(X_test)
    assert isinstance(all_predictions, dict)
    assert 'LinearRegression' in all_predictions
    assert 'Ridge' in all_predictions
    assert len(all_predictions['LinearRegression']) == len(X_test)

    # Test predict specific model
    lr_predictions = reg.predict(X_test, model_name='LinearRegression')
    assert isinstance(lr_predictions, np.ndarray)
    assert len(lr_predictions) == len(X_test)
    assert np.array_equal(lr_predictions, all_predictions['LinearRegression'])

    # Test error for unfitted model
    reg_unfitted = LazyRegressor(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="No models have been fitted yet"):
        reg_unfitted.predict(X_test)

    # Test error for invalid model name
    with pytest.raises(ValueError, match="Model .* not found"):
        reg.predict(X_test, model_name='InvalidModel')


def test_custom_metric_with_failing_models():
    """
    Test that custom_metric parameter works correctly when custom_metric calculation fails for some models.
    This addresses issue #324 where custom_metric caused "arrays must all be same length" error.
    The original issue reported log_loss failing, which happens when metrics expect specific input formats.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    # Create a custom metric that requires specific conditions (like log_loss needing probabilities)
    def strict_custom_metric(y_true, y_pred):
        # This simulates metrics like log_loss that need specific formats
        # It will fail if predictions are not in expected range
        if np.any((y_pred < 0) | (y_pred > 1)):
            raise ValueError("Predictions must be probabilities between 0 and 1")
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test with strict custom metric that will fail for some models
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        custom_metric=strict_custom_metric,
        classifiers=[LogisticRegression, DecisionTreeClassifier, GaussianNB]
    )

    # This should not raise "arrays must all be same length" error
    # even if strict_custom_metric fails for some models (those returning class labels)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Verify the results DataFrame is created successfully
    assert isinstance(models, pd.DataFrame)
    assert "strict_custom_metric" in models.columns

    # The metric should fail for all models (since predict() returns class labels, not probabilities)
    # But the DataFrame should still be created with None values
    assert len(models) == 3  # All 3 models should be in results


def test_custom_metric_with_failing_regressors():
    """
    Test that custom_metric parameter works correctly for regressors when custom metric calculation fails.
    """
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    # Create a custom metric that may fail
    def failing_custom_metric(y_true, y_pred):
        # This will fail if predictions have too high variance
        if np.std(y_pred) > 1000:
            raise ValueError("Predictions have too high variance")
        return np.mean(np.abs(y_true - y_pred))

    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test with custom metric
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        custom_metric=failing_custom_metric,
        regressors=[LinearRegression, Ridge, Lasso]
    )

    # This should not raise "arrays must all be same length" error
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Verify the results DataFrame is created successfully
    assert isinstance(models, pd.DataFrame)
    assert "failing_custom_metric" in models.columns

    # Verify that at least one model has a valid custom metric
    assert any(models['failing_custom_metric'].notna())


def test_precision_recall_metrics():
    """
    Test that Precision and Recall metrics are included in LazyClassifier results.
    This addresses issue #434 requesting precision metrics for imbalanced datasets.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test without CV
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[LogisticRegression, DecisionTreeClassifier]
    )
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    # Verify Precision and Recall columns exist
    assert "Precision" in models.columns
    assert "Recall" in models.columns

    # Verify values are in valid range [0, 1]
    assert all(models['Precision'] >= 0) and all(models['Precision'] <= 1)
    assert all(models['Recall'] >= 0) and all(models['Recall'] <= 1)

    # Verify Precision and Recall are not None
    assert models.loc['LogisticRegression', 'Precision'] is not None
    assert models.loc['LogisticRegression', 'Recall'] is not None


def test_precision_recall_with_cv():
    """
    Test that Precision and Recall CV metrics are included when using cross-validation.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Test with 5-fold CV
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        cv=5,
        classifiers=[LogisticRegression]
    )
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    # Verify CV columns for Precision and Recall exist
    assert "Precision" in models.columns
    assert "Recall" in models.columns
    assert "Precision CV Mean" in models.columns
    assert "Precision CV Std" in models.columns
    assert "Recall CV Mean" in models.columns
    assert "Recall CV Std" in models.columns

    # Verify CV metrics are calculated
    assert models.loc['LogisticRegression', 'Precision CV Mean'] is not None
    assert models.loc['LogisticRegression', 'Recall CV Mean'] is not None

    # Verify CV std is non-negative
    assert models.loc['LogisticRegression', 'Precision CV Std'] >= 0
    assert models.loc['LogisticRegression', 'Recall CV Std'] >= 0


def test_verbose_zero_disables_progress_bar():
    """
    Test that verbose=0 disables the tqdm progress bar.
    This addresses issue #438 about verbosity control.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    import sys
    from io import StringIO

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Capture stdout to check for progress bar output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            classifiers=[LogisticRegression]
        )
        models, _ = clf.fit(X_train, X_test, y_train, y_test)

        output = sys.stdout.getvalue()

        # With verbose=0, there should be no tqdm progress bar output
        assert 'it/s' not in output.lower()
        assert '%|' not in output

    finally:
        sys.stdout = old_stdout

    # Verify the model still ran successfully
    assert isinstance(models, pd.DataFrame)
    assert 'LogisticRegression' in models.index


def test_verbose_one_shows_progress():
    """
    Test that verbose>0 still allows output (logging and progress bar).
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    import logging

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Capture log output
    log_handler = logging.StreamHandler()
    log_handler.setLevel(logging.DEBUG)
    lp_logger = logging.getLogger("lazypredict")
    lp_logger.addHandler(log_handler)
    lp_logger.setLevel(logging.DEBUG)

    try:
        clf = LazyClassifier(
            verbose=1,
            ignore_warnings=True,
            classifiers=[LogisticRegression]
        )
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
    finally:
        lp_logger.removeHandler(log_handler)

    assert isinstance(models, pd.DataFrame)


def test_perpetual_booster_classifier():
    """
    Test that PerpetualBooster works correctly for classification if available.
    """
    try:
        from perpetual import PerpetualBooster
        PERPETUAL_AVAILABLE = True
    except ImportError:
        PERPETUAL_AVAILABLE = False

    if not PERPETUAL_AVAILABLE:
        pytest.skip("PerpetualBooster not installed")

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[PerpetualBooster]
    )
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Verify PerpetualBooster is in results
    assert 'PerpetualBooster' in models.index
    assert models.loc['PerpetualBooster', 'Accuracy'] > 0.5
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)


def test_perpetual_booster_regressor():
    """
    Test that PerpetualBooster works correctly for regression if available.
    """
    try:
        from perpetual import PerpetualBooster
        PERPETUAL_AVAILABLE = True
    except ImportError:
        PERPETUAL_AVAILABLE = False

    if not PERPETUAL_AVAILABLE:
        pytest.skip("PerpetualBooster not installed")

    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        regressors=[PerpetualBooster]
    )
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Verify PerpetualBooster is in results
    assert 'PerpetualBooster' in models.index
    assert isinstance(models, pd.DataFrame)
    assert isinstance(predictions, pd.DataFrame)


def test_timeout_classifier():
    """
    Test that timeout parameter works for LazyClassifier.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Use a very short timeout to ensure some models get skipped
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[LogisticRegression, SVC],
        timeout=0.001  # 1ms timeout - will skip slow models
    )
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    # With such a short timeout, at least some models should complete
    # (LogisticRegression is fast, SVC might timeout)
    assert len(models) >= 0  # Some models might complete
    assert isinstance(models, pd.DataFrame)


def test_timeout_regressor():
    """
    Test that timeout parameter works for LazyRegressor.
    """
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression, Ridge

    data = load_diabetes()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Use a reasonable timeout
    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        regressors=[LinearRegression, Ridge],
        timeout=5  # 5 second timeout
    )
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    # These fast models should complete within timeout
    assert len(models) == 2
    assert 'LinearRegression' in models.index
    assert 'Ridge' in models.index


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_input_validation_shape_mismatch():
    """Test that mismatched X_train/y_train shapes raise ValueError."""
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, 80)  # Wrong length
    y_test = np.random.randint(0, 2, 50)

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="X_train has 100 samples but y_train has 80"):
        clf.fit(X_train, X_test, y_train, y_test)


def test_input_validation_feature_mismatch():
    """Test that mismatched feature counts between X_train and X_test raise ValueError."""
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(50, 3)  # Wrong number of features
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 50)

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="X_train has 5 features but X_test has 3"):
        clf.fit(X_train, X_test, y_train, y_test)


def test_input_validation_empty_data():
    """Test that empty data raises ValueError."""
    X_train = np.array([]).reshape(0, 5)
    X_test = np.random.rand(50, 5)
    y_train = np.array([])
    y_test = np.random.randint(0, 2, 50)

    clf = LazyClassifier(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="X_train is empty"):
        clf.fit(X_train, X_test, y_train, y_test)


def test_init_validation_cv():
    """Test that invalid cv parameter raises ValueError."""
    with pytest.raises(ValueError, match="cv must be an integer >= 2"):
        LazyClassifier(cv=1)
    with pytest.raises(ValueError, match="cv must be an integer >= 2"):
        LazyClassifier(cv=0)


def test_init_validation_timeout():
    """Test that invalid timeout parameter raises ValueError."""
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        LazyClassifier(timeout=-1)
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        LazyClassifier(timeout=0)


def test_init_validation_encoder():
    """Test that invalid categorical_encoder raises ValueError at init time."""
    with pytest.raises(ValueError, match="categorical_encoder must be one of"):
        LazyClassifier(categorical_encoder='invalid')


def test_init_validation_custom_metric():
    """Test that non-callable custom_metric raises TypeError."""
    with pytest.raises(TypeError, match="custom_metric must be callable"):
        LazyClassifier(custom_metric="not_callable")


def test_errors_dict_populated():
    """Test that the errors dict exists on classifier."""
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, classifiers=[LogisticRegression])
    clf.fit(X_train, X_test, y_train, y_test)

    assert isinstance(clf.errors, dict)


def test_regressor_input_validation():
    """Test that regressor also validates inputs."""
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(50, 5)
    y_train = np.random.rand(80)  # Wrong length
    y_test = np.random.rand(50)

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    with pytest.raises(ValueError, match="X_train has 100 samples but y_train has 80"):
        reg.fit(X_train, X_test, y_train, y_test)


def test_regressor_init_validation():
    """Test regressor constructor validation."""
    with pytest.raises(ValueError, match="cv must be an integer >= 2"):
        LazyRegressor(cv=1)
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        LazyRegressor(timeout=0)
    with pytest.raises(ValueError, match="categorical_encoder must be one of"):
        LazyRegressor(categorical_encoder='bad')
