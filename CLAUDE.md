# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LazyPredict is an automated machine learning library that trains and evaluates 40+ classification and regression models with minimal code. It provides quick model comparison to identify promising algorithms without manual parameter tuning.

**Key Features:**
- Automated model selection for classification and regression
- Built-in MLflow integration for experiment tracking
- SHAP-based model explainability (optional)
- CLI interface for quick experiments
- Support for Python 3.9-3.14

**Python Version Support:**
- Python 3.9-3.14: Fully supported with modern dependencies
- Python 3.8: End-of-life (October 2024) - No longer supported
- Recommended: Python 3.11 or 3.12 for best stability and performance

## Development Commands

### Setup
```bash
# Install in development mode
python setup.py develop
pip install -r requirements_dev.txt
```

### Testing

**Dependency Installation by Python Version:**
```bash
# Python 3.9-3.14 (Modern - Recommended)
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Verify Python version compatibility
python --version  # Should be 3.9+
```

**Run Tests:**
```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_supervised.py -v

# Run comprehensive test suite (40+ test cases)
pytest tests/test_supervised_comprehensive.py -v

# Run specific test
pytest tests/test_supervised_comprehensive.py::TestLazyClassifier::test_parallel_training -v

# Run with coverage
pytest --cov=lazypredict --cov-report=html --cov-report=term -v

# Run tests in parallel (faster - recommended for CI/local development)
pytest -n auto -v

# Run with timeout protection (300s per test)
pytest -v --timeout=300

# Stop at first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Linting
```bash
flake8 lazypredict tests
```

### Building & Distribution
```bash
# Clean build artifacts
make clean

# Build distribution packages
make dist

# Install locally
make install
```

### Documentation
```bash
# Generate and view documentation
make docs
```

## Architecture

### Core Components

**1. `lazypredict/Supervised.py`** (733 lines)
- **`BaseLazyEstimator`**: Abstract base class implementing shared logic for lazy estimators
  - Handles data preprocessing (numeric/categorical features)
  - Manages parallel/sequential training (`_train_single_model`, `_train_parallel`, `_train_sequential`)
  - MLflow integration (`setup_mlflow()`, `is_mlflow_tracking_enabled()`)
  - Model storage in `self.trained_models` dict

- **`LazyClassifier`**: Classification model selection
  - Trains 40+ classifiers (sklearn + XGBoost + LightGBM)
  - Metrics: Accuracy, Balanced Accuracy, ROC AUC, F1 Score
  - Sorted by Balanced Accuracy

- **`LazyRegressor`**: Regression model selection
  - Trains 40+ regressors (sklearn + XGBoost + LightGBM)
  - Metrics: R-Squared, Adjusted R-Squared, RMSE
  - Sorted by Adjusted R-Squared

- **Key Parameters:**
  - `verbose`: Control logging output (0=quiet, >0=detailed)
  - `ignore_warnings`: Suppress model warnings (default: True)
  - `custom_metric`: User-defined evaluation function
  - `predictions`: Return predictions DataFrame (default: False)
  - `random_state`: Reproducibility seed (default: 42)
  - `n_jobs`: Parallel training (-1 for all CPUs, default: 1)
  - `classifiers`/`regressors`: "all" or list of specific models

**2. `lazypredict/Explainer.py`** (582 lines)
- **`ModelExplainer`**: SHAP-based model explainability
  - Requires `shap` package (optional dependency)
  - Works with trained LazyClassifier/LazyRegressor instances
  - Methods:
    - `feature_importance()`: Global feature importance
    - `plot_summary()`: SHAP summary plots (dot/bar/violin)
    - `explain_prediction()`: Single prediction waterfall plots
    - `plot_dependence()`: Feature dependence plots
    - `compare_models()`: Cross-model feature importance comparison
    - `get_top_features()`: Top features for specific predictions
  - Automatically selects appropriate explainer (TreeExplainer, LinearExplainer, KernelExplainer)

**3. `lazypredict/cli.py`** (202 lines)
- Click-based CLI interface
- Commands:
  - `lazypredict classify <file> --target <column>`: Run classification
  - `lazypredict regress <file> --target <column>`: Run regression
  - `lazypredict info`: Display package information
- Options: `--test-size`, `--output`, `--predictions`, `--n-jobs`, `--random-state`, `--verbose`

### Data Preprocessing Pipeline

The preprocessing pipeline handles mixed data types automatically:

- **Numeric features**: SimpleImputer (mean) → StandardScaler
- **Categorical features (low cardinality <11)**: SimpleImputer (constant) → OneHotEncoder
- **Categorical features (high cardinality ≥11)**: SimpleImputer (constant) → OrdinalEncoder

Helper function `get_card_split(df, cols, n=11)` determines cardinality split.

### Model Lists

**Removed Classifiers:** ClassifierChain, ComplementNB, GradientBoostingClassifier, GaussianProcessClassifier, HistGradientBoostingClassifier, MLPClassifier, LogisticRegressionCV, MultiOutputClassifier, MultinomialNB, OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier, RadiusNeighborsClassifier, VotingClassifier

**Removed Regressors:** TheilSenRegressor, ARDRegression, CCA, IsotonicRegression, StackingRegressor, MultiOutputRegressor, MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV, PLSCanonical, PLSRegression, RadiusNeighborsRegressor, RegressorChain, VotingRegressor

**Added Models:** XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor

### MLflow Integration

MLflow tracking is enabled via environment variable:
```python
import os
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
```

When enabled, automatically logs:
- Model parameters
- Metrics (accuracy, RMSE, etc.)
- Training time
- Model artifacts with signatures
- Registered models with naming: `{lazyclassifier|lazyregressor}_{model_name}`

Check: `is_mlflow_tracking_enabled()` returns True if `MLFLOW_TRACKING_URI` is set.

### Training Modes

**Sequential Training (n_jobs=1):**
- Uses tqdm progress bar (or notebook_tqdm in Jupyter)
- Trains models one at a time

**Parallel Training (n_jobs>1 or n_jobs=-1):**
- Uses joblib.Parallel with delayed execution
- Trains multiple models concurrently
- Note: MLflow runs are managed per-thread

## Testing Strategy

**Test Files:**
- `tests/test_supervised.py`: Original test suite with MLflow cleanup
- `tests/test_supervised_comprehensive.py`: Comprehensive suite with 40+ parametrized tests
- `tests/test_helpers.py`: Helper function tests
- `tests/test_cli.py`: CLI tests
- `tests/test_init.py`: Package initialization tests
- `tests/test_lazypredict.py`: Additional integration tests

**Key Fixtures:**
- `cleanup_mlflow` (autouse): Cleans up MLflow DB/artifacts before/after tests
- `classification_data`: Breast cancer dataset split
- `regression_data`: Diabetes dataset split
- `categorical_classification_data`: Synthetic categorical data

**Parametrized Tests:**
- Parallel training: `@pytest.mark.parametrize("n_jobs", [1, 2, -1])`
- Multiclass: `@pytest.mark.parametrize("n_classes", [2, 3, 5])`
- Cardinality: `@pytest.mark.parametrize("n", [5, 10, 15, 20])`

**Coverage Target:** 75-85% overall, 80-90% for Supervised.py

See `TESTING_GUIDE.md` for comprehensive testing documentation.

## Common Patterns

### Basic Usage Pattern
```python
from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Classification
clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=-1)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Regression
reg = LazyRegressor(verbose=0, ignore_warnings=True, n_jobs=-1)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Access trained models
trained_models = clf.provide_models(X_train, X_test, y_train, y_test)
best_model = trained_models['LogisticRegression']
```

### Custom Metrics
```python
def custom_metric(y_true, y_pred):
    return some_metric(y_true, y_pred)

clf = LazyClassifier(custom_metric=custom_metric)
```

### Model Explainability
```python
from lazypredict.Explainer import ModelExplainer

explainer = ModelExplainer(clf, X_train, X_test)
importance = explainer.feature_importance('LogisticRegression', top_n=10)
explainer.plot_summary('LogisticRegression')
explainer.explain_prediction('LogisticRegression', instance_idx=0)
```

## Important Implementation Details

1. **Trained Models Storage**: All fitted models are stored in `self.trained_models` as scikit-learn Pipeline objects with preprocessing + model steps.

2. **Error Handling**: Failed models return `None` from `_train_single_model()` and are excluded from results. Controlled by `ignore_warnings` parameter.

3. **Random State**: Models that support `random_state` parameter receive it automatically for reproducibility.

4. **Data Conversion**: NumPy arrays are automatically converted to pandas DataFrames with auto-generated column names.

5. **MLflow Runs**: Each model gets its own MLflow run. Runs are properly closed even on exceptions.

6. **SHAP Explainer Selection**: ModelExplainer automatically chooses TreeExplainer, LinearExplainer, or KernelExplainer based on model type.

7. **Multiclass Classification**: SHAP values for binary classification use positive class (index 1). For multiclass, handles list of arrays.

## File References

When referencing code locations for debugging or changes:
- Core training logic: `lazypredict/Supervised.py:301-379` (`_train_single_model`)
- Parallel training: `lazypredict/Supervised.py:488-506` (`_train_parallel`)
- Preprocessing: `lazypredict/Supervised.py:257-271` (`_get_preprocessor`)
- MLflow setup: `lazypredict/Supervised.py:172-186` (`setup_mlflow`)
- SHAP explainer selection: `lazypredict/Explainer.py:122-187` (`_get_explainer`)
- CLI commands: `lazypredict/cli.py:23-181`

## Development Notes

- The codebase uses type hints extensively (see imports from `typing`)
- Logging is configured via Python's `logging` module, not print statements
- Progress bars adapt to Jupyter environments (tqdm vs notebook_tqdm)
- Matplotlib backend set to 'Agg' to avoid tkinter threading issues (line 12 in Supervised.py)
- Backward compatibility aliases: `Regression = LazyRegressor`, `Classification = LazyClassifier`
