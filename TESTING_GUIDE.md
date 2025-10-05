# Testing Guide for LazyPredict

## Overview

This document provides comprehensive information about testing LazyPredict, including test structure, execution, and expected coverage.

## Test Files

### 1. `tests/test_supervised.py`
**Original test suite** with MLflow cleanup fixtures.

**Coverage:**
- Basic LazyClassifier functionality
- Basic LazyRegressor functionality
- MLflow integration
- Test isolation with cleanup fixtures

**Run command:**
```bash
pytest tests/test_supervised.py -v
```

### 2. `tests/test_supervised_comprehensive.py` ⭐ NEW
**Comprehensive test suite** with parametrization and edge cases.

**Coverage:**
- ✅ **LazyClassifier Tests** (7 tests)
  - Basic fit
  - Predictions output
  - Custom metrics
  - Parallel training (n_jobs=1, 2, -1)
  - Specific classifier selection
  - Categorical features handling
  - Model retrieval via provide_models()
  - MLflow integration

- ✅ **LazyRegressor Tests** (6 tests)
  - Basic fit
  - Predictions output
  - Custom metrics
  - Parallel training (n_jobs=1, 2, -1)
  - Specific regressor selection
  - Model retrieval via provide_models()

- ✅ **Edge Cases** (8 tests)
  - Empty dataset (error handling)
  - Single feature dataset
  - All categorical features
  - Multiclass classification (2, 3, 5 classes)
  - Missing values handling
  - Invalid classifier list
  
- ✅ **Helper Functions** (3 tests)
  - get_card_split() basic functionality
  - get_card_split() with different thresholds (n=5, 10, 15, 20)
  - get_card_split() with empty input

- ✅ **Verbosity & Warnings** (2 tests)
  - Verbose mode (verbose=2)
  - Warnings not ignored (ignore_warnings=False)

- ✅ **Random State** (2 tests)
  - Reproducibility with same random_state
  - Different results with different random_states

**Total:** 40+ test cases with comprehensive parametrization

**Run command:**
```bash
pytest tests/test_supervised_comprehensive.py -v
```

### 3. `tests/test_helpers.py`
Tests for helper functions and utilities.

### 4. `tests/test_cli.py`
Tests for CLI functionality.

### 5. `tests/test_init.py`
Tests for package initialization.

### 6. `tests/test_lazypredict.py`
Additional integration tests.

## Running Tests

### Run All Tests
```bash
pytest -v
```

### Run Specific Test File
```bash
pytest tests/test_supervised_comprehensive.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_supervised_comprehensive.py::TestLazyClassifier -v
```

### Run Specific Test
```bash
pytest tests/test_supervised_comprehensive.py::TestLazyClassifier::test_parallel_training -v
```

### Run with Coverage
```bash
pytest --cov=lazypredict --cov-report=html --cov-report=term -v
```

### Run with Coverage (Comprehensive Test Only)
```bash
pytest tests/test_supervised_comprehensive.py --cov=lazypredict.Supervised --cov-report=html --cov-report=term -v
```

### Run Tests in Parallel (faster)
```bash
pytest -n auto -v
```

## Test Fixtures

### `cleanup_mlflow` (autouse=True)
Automatically cleans up MLflow artifacts before and after each test:
- Removes `mlflow.db` database
- Removes `mlruns` directory
- Ends any active MLflow runs
- Ensures test isolation

### `classification_data`
Provides breast cancer dataset split into train/test sets.

### `regression_data`
Provides diabetes dataset split into train/test sets.

### `categorical_classification_data`
Provides synthetic dataset with categorical features for testing preprocessing.

## Parametrized Tests

The comprehensive test suite uses `@pytest.mark.parametrize` for efficient testing:

### Parallel Training (n_jobs)
```python
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_parallel_training(self, classification_data, n_jobs):
    # Tests sequential (n_jobs=1), dual-core (n_jobs=2), 
    # and all-cores (n_jobs=-1) training
```

### Multiclass Classification
```python
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_multiclass_classification(self, n_classes):
    # Tests binary, 3-class, and 5-class classification
```

### Cardinality Threshold
```python
@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_get_card_split_different_n(self, n):
    # Tests categorical feature cardinality detection
```

## Expected Coverage

### Current Coverage (Estimated)
- **Overall:** ~45-50%
- **Supervised.py:** ~40-45%
- **cli.py:** ~0% (needs CLI-specific tests)

### Target Coverage with Comprehensive Tests
- **Overall:** 75-85%
- **Supervised.py:** 80-90%
  - LazyClassifier: 85%+
  - LazyRegressor: 85%+
  - BaseLazyEstimator: 90%+
  - Helper functions: 95%+
- **cli.py:** 70-80% (with CLI tests)

### Coverage Report Location
After running with `--cov-report=html`:
```
htmlcov/index.html
```

Open in browser:
```bash
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

## Test Execution in CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

```yaml
- name: Test with pytest
  run: |
    pytest --cov=lazypredict --cov-report=xml --cov-report=html --cov-report=term -v
```

Coverage is automatically uploaded to Codecov.

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestLazyClassifier`)
- Test methods: `test_*` (e.g., `test_parallel_training`)

### Test Structure Template
```python
import pytest
from lazypredict.Supervised import LazyClassifier

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        clf = LazyClassifier(verbose=0)
        
        # Act
        result = clf.fit(X_train, X_test, y_train, y_test)
        
        # Assert
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.parametrize("param", [1, 2, 3])
    def test_parametrized(self, param):
        """Test with different parameters."""
        # Test implementation
        pass
```

### Best Practices
1. **Use fixtures** for common test data
2. **Parametrize** similar tests with different inputs
3. **Test edge cases** (empty, single item, large datasets)
4. **Test error handling** with `pytest.raises()`
5. **Keep tests isolated** (use autouse cleanup fixtures)
6. **Document tests** with clear docstrings
7. **Assert meaningful things** (not just "doesn't crash")

## Test Categories

### Unit Tests
- Individual functions and methods
- Isolated from external dependencies
- Fast execution (<1s per test)

### Integration Tests
- Multiple components working together
- MLflow integration
- Pipeline functionality

### Regression Tests
- Ensure bugs don't reappear
- Edge cases that previously failed

### Performance Tests
- Parallel training speed
- Large dataset handling
- Memory usage

## Debugging Failed Tests

### View Full Traceback
```bash
pytest tests/test_supervised_comprehensive.py -v --tb=long
```

### Stop at First Failure
```bash
pytest tests/test_supervised_comprehensive.py -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

### Run Failed Tests First
```bash
pytest --ff
```

### Verbose Output with Print Statements
```bash
pytest tests/test_supervised_comprehensive.py -v -s
```

### Debug with PDB
```bash
pytest tests/test_supervised_comprehensive.py --pdb
```

## Test Matrix (CI/CD)

The project tests against multiple Python versions:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13

Ensure compatibility across all versions.

## Known Test Limitations

1. **MLflow Cleanup**: Some tests may leave `mlflow.db` if interrupted
   - **Solution**: Manual cleanup or use cleanup fixture
   
2. **Parallel Test Execution**: May cause database locking issues
   - **Solution**: Run comprehensive tests sequentially

3. **Long Execution Time**: Full test suite takes ~2-5 minutes
   - **Solution**: Use `pytest -n auto` for parallel execution
   
4. **Memory Usage**: Tests with large models may use significant RAM
   - **Solution**: Limit concurrent tests or use smaller datasets

## Test Maintenance

### Regular Updates Needed
- ✅ Add tests for new features
- ✅ Update tests when API changes
- ✅ Add regression tests for reported bugs
- ✅ Maintain >75% code coverage
- ✅ Ensure all tests pass before merging

### Test Review Checklist
- [ ] All tests pass locally
- [ ] Coverage increased or maintained
- [ ] New features have tests
- [ ] Edge cases covered
- [ ] Documentation updated
- [ ] CI/CD passes all checks

## Additional Resources

- **pytest documentation**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **Codecov**: https://docs.codecov.io/

## Summary

The comprehensive test suite provides:
- ✅ **40+ test cases** covering core functionality
- ✅ **Parametrized tests** for efficiency
- ✅ **Edge case testing** for robustness
- ✅ **Parallel training validation**
- ✅ **Custom metric support testing**
- ✅ **Error handling verification**
- ✅ **Target 80%+ coverage** for Supervised.py

Run tests regularly during development and before committing changes!
