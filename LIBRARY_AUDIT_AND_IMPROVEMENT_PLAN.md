# LazyPredict Library Audit & Improvement Plan

**Audit Date:** 2026-02-28
**Version Audited:** 0.2.16
**Codebase Size:** ~1,342 lines (core), ~750+ lines (tests), 14 Python files

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architect's Perspective](#1-architects-perspective)
3. [Developer's Perspective](#2-developers-perspective)
4. [Security Expert's Perspective](#3-security-experts-perspective)
5. [Prioritized Improvement Plan](#4-prioritized-improvement-plan)
6. [Implementation Roadmap](#5-implementation-roadmap)

---

## Executive Summary

LazyPredict is a useful prototyping tool that trains 40+ ML models with minimal code. However, the current codebase has significant structural, quality, and robustness issues that limit its reliability for production-adjacent workflows. This audit identifies **47 distinct issues** across three expert perspectives and proposes a phased improvement plan for the next release.

**Critical finding:** The entire library lives in a single 1,342-line file (`Supervised.py`) with ~400 lines of duplicated code between `LazyClassifier` and `LazyRegressor`, no type hints, no logging framework, broad exception swallowing, global state mutation, and incomplete input validation.

---

## 1. Architect's Perspective

### 1.1 Package Structure — POOR

| Issue | Severity | Location |
|-------|----------|----------|
| **Monolithic file**: All logic in a single `Supervised.py` (1,342 lines) | High | `Supervised.py` |
| **No module separation**: Preprocessing, model registry, metrics, and MLflow integration are all interleaved | High | `Supervised.py` |
| **No `__all__` exports**: Public API surface is undefined; users can import internals | Medium | `__init__.py` |
| **CLI is a placeholder**: Entry point registered but does nothing useful | Low | `cli.py` |
| **Aliases at module bottom**: `Regression = LazyRegressor` / `Classification = LazyClassifier` are undocumented | Low | `Supervised.py:1341-1342` |

**Recommended structure:**
```
lazypredict/
├── __init__.py          # Public API exports (__all__)
├── _base.py             # Abstract base class (LazyEstimator)
├── classifier.py        # LazyClassifier
├── regressor.py         # LazyRegressor
├── preprocessing.py     # Encoders, transformers, get_card_split
├── registry.py          # CLASSIFIERS, REGRESSORS lists + model discovery
├── metrics.py           # adjusted_rsquared, metric aggregation
├── integrations/
│   └── mlflow.py        # MLflow setup, logging helpers
├── config.py            # Constants, defaults, removed model lists
├── exceptions.py        # TimeoutException and custom errors
├── _compat.py           # Platform-specific (Windows timeout), optional deps
└── cli.py               # Real CLI or remove entry point
```

### 1.2 Code Duplication — CRITICAL (~400 lines)

`LazyClassifier` and `LazyRegressor` share **80%+ identical code**:

| Shared Pattern | Classifier Lines | Regressor Lines |
|----------------|-----------------|-----------------|
| `__init__` method | 367-390 | 960-983 |
| Input conversion (ndarray→DataFrame) | 448-450 | 1033-1035 |
| Feature type detection | 452-454 | 1037-1039 |
| Preprocessing pipeline setup | 456-470 | 1041-1055 |
| Model list normalization | 472-483 | 1057-1068 |
| Model training loop structure | 487-693 | 1072-1224 |
| `provide_models` method | 785-812 | 1264-1291 |
| `predict` method | 814-859 | 1293-1338 |

**Fix:** Create a `LazyEstimator` base class with shared logic; subclasses override only metrics computation and model-type-specific behavior.

### 1.3 Dependency Architecture — FRAGILE

| Issue | Detail |
|-------|--------|
| Hard dependencies on `xgboost` and `lightgbm` | Lines 53, 56 — fail at import time if not installed |
| `mlflow>=2.0.0` in requirements.txt | Heavy dependency (300+ MB) for an optional feature |
| `pytest-runner` in production requirements | Should be dev-only |
| No version pins | Any `scikit-learn`, `pandas`, `xgboost` update can break the library |
| `category_encoders` optional but undocumented | Only discovered at runtime when user picks `'target'` or `'binary'` encoder |

**Fix:**
- Move `xgboost`, `lightgbm`, `mlflow` to optional extras: `pip install lazypredict[xgboost,mlflow]`
- Pin minimum versions for core deps
- Remove `pytest-runner` from production requirements
- Use `importlib.metadata` or try/except for all optional deps consistently

### 1.4 API Design Issues

| Issue | Detail | Severity |
|-------|--------|----------|
| `fit()` requires both train AND test data | Unusual API — sklearn convention is `fit(X, y)`, predict on test separately | High |
| `fit()` return type changes based on `predictions` flag | Returns `(scores, predictions_df)` tuple or just `scores` — violates consistent return types | High |
| `provide_models()` re-fits if not already fitted | Surprising side-effect — should raise instead | Medium |
| `predictions` parameter name conflicts with local `predictions` dict | Variable shadowing in `fit()` at lines 428 vs 678 | Medium |
| `verbose=0` hides ALL output including errors | No way to see errors without also seeing progress bars | Medium |
| Timeout doesn't actually timeout | Lines 509-518: Model trains fully, THEN checks if time exceeded | High |

### 1.5 Versioning & Build

| Issue | Detail |
|-------|--------|
| Version defined in 3 places | `setup.py:54`, `__init__.py:7`, `meta.yaml` (still 0.2.15) |
| `.bumpversion.cfg` only updates 2 of 3 locations | `meta.yaml` not included |
| No `pyproject.toml` | Modern packaging standard not adopted |
| `setup_requires=["pytest-runner"]` is deprecated | Should use `pyproject.toml` build-system requires |

### 1.6 Test Architecture

| Issue | Detail |
|-------|--------|
| CI uses `pytest \|\| true` | **Tests never fail the build** (line 50 of ci.yml) |
| No coverage reporting | No `pytest-cov` or coverage thresholds |
| No negative tests | No tests for invalid inputs, error paths, edge cases |
| Tests import private symbols | `from lazypredict.Supervised import ... INTEL_EXTENSION_AVAILABLE` |
| No test for Windows timeout bypass | Platform-specific code paths untested |
| No integration test isolation | MLflow tests depend on environment variables |

---

## 2. Developer's Perspective

### 2.1 Missing Type Hints — ZERO type annotations

**Every** function and method lacks type hints. This means:
- No IDE autocomplete for users
- No static analysis (mypy) possible
- API is unclear without reading source

**Example (current):**
```python
def fit(self, X_train, X_test, y_train, y_test):
```

**Should be:**
```python
def fit(
    self,
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
```

### 2.2 Logging — ALL print() statements (26+ instances)

| Line(s) | Context | What's printed |
|----------|---------|---------------|
| 208-209, 215-216 | Encoder fallback warnings | Warning messages |
| 482-483 | Invalid classifiers | Exception + "Invalid Classifier(s)" |
| 515 | Timeout exceeded | Skip message |
| 565-566 | CV failure | Cross-validation error |
| 607-608 | ROC AUC failure | Exception details |
| 626 | MLflow model logging | Error message |
| 647-648 | Custom metric failure | Exception details |
| 652-677 | Verbose output | Full metric dictionaries |
| 691-692 | Model failure | "model failed to execute" |
| 1067-1068 | Invalid regressors | Exception + "Invalid Regressor(s)" |
| 1100 | Timeout exceeded | Skip message |
| 1143-1144 | CV failure | Cross-validation error |
| 1175 | MLflow model logging | Error message |
| 1194-1195 | Custom metric failure | Exception details |
| 1209 | Verbose output | Metric dictionary |
| 1223-1224 | Model failure | "model failed to execute" |

**Fix:** Replace all with Python's `logging` module. Users can configure log levels, handlers, and formatters.

### 2.3 Exception Handling — DANGEROUS

| Issue | Location | Severity |
|-------|----------|----------|
| **Bare `except:`** catches KeyboardInterrupt, SystemExit | Line 20 | Critical |
| **Bare `except:`** swallows ROC AUC errors | Line 560 | High |
| Broad `except Exception` in 10+ places | Lines 481, 563, 604, 624, 643, 685, 1066, 1141, 1173, 1190, 1217 | High |
| Exceptions printed then silently continued | All error handlers | Medium |
| No structured error collection | Models fail without trace | Medium |

**Fix:**
- Replace bare `except:` with `except Exception:`  (or more specific)
- Collect errors in a structured `self.errors: Dict[str, Exception]` dict
- Log with proper severity levels
- Expose errors to users via `clf.errors` property

### 2.4 Global State Mutation

| Line | Code | Impact |
|------|------|--------|
| 96 | `warnings.filterwarnings("ignore")` | Suppresses ALL warnings for entire Python process |
| 97 | `pd.set_option("display.precision", 2)` | Changes pandas display for all user code |
| 98 | `pd.set_option("display.float_format", ...)` | Changes pandas float format globally |

**Fix:** Remove global state changes. Apply formatting only to returned DataFrames using `.style` or context managers.

### 2.5 Hardcoded Values & Magic Numbers

| Value | Location | Description |
|-------|----------|-------------|
| `n=11` | `get_card_split()` line 266 | Cardinality split threshold |
| `"%.2f"` | Line 98 | Float display precision |
| `"missing"` | Line 166 | Fill value for categorical NAs |
| `-1` | Lines 200, 203, 205 | Unknown category encoding value |
| `2` | Line 592 | Binary classification detection (`shape[1] == 2`) |
| `42` | Line 373 | Default random_state |
| `n_jobs=-1` | Lines 539, 1119 | Use all CPU cores for CV |

**Fix:** Extract to a `Config` dataclass or module-level constants with documentation.

### 2.6 Import Issues

| Issue | Location |
|-------|----------|
| `datetime` imported but never used | Line 23 |
| `sys` imported but never used | Line 8 |
| `MissingIndicator` imported but never used | Line 27 |
| Imports not PEP 8 grouped | Standard lib split across two sections |
| `xgboost` and `lightgbm` are hard imports that fail loudly | Lines 53, 56 |

### 2.7 Code Smells

| Smell | Location | Detail |
|-------|----------|--------|
| Uppercase variable names for lists | Lines 420-427 | `Accuracy`, `B_Accuracy`, `ROC_AUC` etc. — not constants |
| Inconsistent naming | `B_Accuracy` vs `b_accuracy` | Mix of styles |
| Long method (`fit()`) | 300+ lines each | Should be decomposed |
| Conditional CV list initialization | Lines 431-443 | Lists created inside `if self.cv:` block — fragile |
| Duplicate DataFrame construction | Lines 694-776, 1226-1258 | 4 nearly identical blocks for cv/no-cv × custom-metric/no-custom-metric |
| Comment says "Helper class for performing classification" above `LazyRegressor` | Line 866 | Copy-paste error |

### 2.8 Docstring Issues

| Issue | Location |
|-------|----------|
| `adjusted_rsquared()` has no docstring | Line 862 |
| `is_mlflow_tracking_enabled()` docstring is minimal | Line 81 |
| y_train/y_test docstrings say "columns is the number of features" | Lines 407-411 — copy-paste error |
| LazyClassifier docstring shows models that are in `removed_classifiers` | Lines 337-340 |
| No module-level docstring explaining the file structure | Top of Supervised.py |

---

## 3. Security Expert's Perspective

### 3.1 Input Validation — HIGH RISK

| Missing Validation | Impact | Location |
|--------------------|--------|----------|
| No check that X_train/X_test have same columns | Silent incorrect results or crash | `fit()` |
| No check that y_train length matches X_train rows | Cryptic sklearn error | `fit()` |
| No check for empty DataFrames | Unclear error messages | `fit()` |
| No type validation on constructor parameters | TypeError deep in pipeline | `__init__()` |
| `cv` accepts any integer including 0, 1, negative | `cross_validate` will crash | `__init__()` |
| `timeout` accepts negative values | Undefined behavior | `__init__()` |
| `categorical_encoder` not validated until `get_categorical_encoder()` | Late failure | `fit()` |
| ndarray→DataFrame conversion loses column names | Lines 448-450: `pd.DataFrame(X_train)` — no columns | `fit()` |

### 3.2 Resource Exhaustion — HIGH RISK

| Risk | Detail | Location |
|------|--------|----------|
| **Timeout doesn't actually prevent long runs** | Model trains completely, timeout checked AFTER fit | Lines 509-518 |
| **Windows has no timeout at all** | `signal.SIGALRM` unavailable; silently yields | Lines 252-263 |
| **`n_jobs=-1` uses all CPU cores** | CV can make system unresponsive | Lines 539, 1119 |
| **No memory limits** | OneHotEncoder on high-cardinality features can OOM | Preprocessing |
| **No model count limit** | 40+ models trained sequentially with no cap | Model loop |
| **No dataset size checks** | Very large datasets will cause long runs | `fit()` |

### 3.3 Information Leakage

| Risk | Detail | Location |
|------|--------|----------|
| Raw exception objects printed to stdout | May contain file paths, system info | Lines 482, 608, 692 |
| MLflow errors may expose tracking URIs | Connection strings in error messages | Line 626 |
| Timing information reveals machine capabilities | `{fit_time:.2f}s` in skip messages | Line 515 |
| Training data shapes logged to MLflow | `infer_signature(X_train, ...)` | Line 621 |

### 3.4 Dependency Security

| Issue | Risk | Detail |
|-------|------|--------|
| **No pinned versions** in requirements.txt | Supply chain | Any breaking update or compromised version pulled |
| **No lock file** (requirements.lock / poetry.lock) | Reproducibility | Cannot reproduce exact environment |
| **`mlflow>=2.0.0`** only lower-bounded | Breaking changes | MLflow 3.x could break integration |
| **`pytest-runner`** in production deps | Unnecessary attack surface | Test dependency in production |
| **xgboost/lightgbm** unpinned | C++ backend compatibility | Version mismatches cause segfaults |

### 3.5 Thread Safety & Race Conditions

| Issue | Detail | Location |
|-------|--------|----------|
| `signal.signal()` is process-global | Concurrent timeout contexts overwrite each other | Lines 254-255 |
| `CLASSIFIERS`/`REGRESSORS` are mutable module-level lists | Can be modified during iteration | Lines 135-157 |
| `warnings.filterwarnings("ignore")` is global | Affects all threads | Line 96 |
| MLflow global tracking URI | Multiple instances share state | Line 91 |
| `self.classifiers = CLASSIFIERS` mutates instance reference to shared list | Line 473 | `fit()` |

### 3.6 Exception Handling Security

| Issue | Detail |
|-------|--------|
| Bare `except:` at line 20 catches `SystemExit` and `KeyboardInterrupt` | User cannot Ctrl+C during IPython detection |
| Silent failure mode is default (`ignore_warnings=True`) | Security-relevant errors are hidden |
| No audit trail of failed models | Cannot investigate failures post-hoc |
| `print(exception)` may expose internal state | Should use sanitized error messages |

### 3.7 Safe Practices (What's Already Good)

- No `eval()`, `exec()`, or `compile()` usage
- No pickle deserialization of untrusted data
- No SQL operations or file path manipulation from user input
- No network calls except through MLflow (user-configured)
- sklearn's `all_estimators()` is a safe model discovery mechanism

---

## 4. Prioritized Improvement Plan

### Phase 1: Critical Fixes (Week 1-2)

These fix correctness and safety bugs without changing the API:

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 1.1 | Replace bare `except:` with `except Exception:` at lines 20, 560 | Security | Critical | 10 min |
| 1.2 | Remove global `warnings.filterwarnings("ignore")` and `pd.set_option` calls | Security/Dev | High | 30 min |
| 1.3 | Add input validation in `__init__()` for `cv`, `timeout`, `categorical_encoder` | Security | High | 1 hr |
| 1.4 | Add input validation in `fit()` for X/y shape compatibility | Security | High | 1 hr |
| 1.5 | Fix timeout to use `threading.Timer` for actual interruption (cross-platform) | Security | High | 2 hr |
| 1.6 | Remove unused imports (`datetime`, `sys`, `MissingIndicator`) | Developer | Low | 10 min |
| 1.7 | Remove `pytest-runner` from production requirements | Security | Medium | 5 min |
| 1.8 | Fix `meta.yaml` version (0.2.15 → 0.2.16) | Developer | Low | 5 min |
| 1.9 | Fix CI: remove `\|\| true` from pytest command | Architect | High | 5 min |
| 1.10 | Fix copy-paste error: "Helper class for performing classification" above LazyRegressor | Developer | Low | 5 min |

### Phase 2: Logging & Error Handling (Week 2-3)

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 2.1 | Add `logging` module with `lazypredict` logger | Developer | High | 2 hr |
| 2.2 | Replace all 26+ `print()` calls with appropriate log levels | Developer | High | 2 hr |
| 2.3 | Add `self.errors: Dict[str, Exception]` to collect model failures | Developer | High | 1 hr |
| 2.4 | Use specific exception types instead of broad `except Exception` | Security | Medium | 2 hr |
| 2.5 | Add structured error messages (no raw exception objects) | Security | Medium | 1 hr |

### Phase 3: Type Hints & Documentation (Week 3-4)

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 3.1 | Add type hints to all public methods and functions | Developer | High | 4 hr |
| 3.2 | Add `py.typed` marker file for PEP 561 | Developer | Medium | 5 min |
| 3.3 | Fix incorrect docstrings (y_train/y_test descriptions) | Developer | Medium | 30 min |
| 3.4 | Add docstring to `adjusted_rsquared()` | Developer | Low | 10 min |
| 3.5 | Update LazyClassifier docstring example (remove listed-but-removed models) | Developer | Low | 30 min |
| 3.6 | Add module-level docstring to `Supervised.py` | Developer | Low | 15 min |

### Phase 4: Architecture Refactor (Week 4-6)

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 4.1 | Create `LazyEstimator` base class extracting shared logic | Architect | High | 8 hr |
| 4.2 | Split `Supervised.py` into separate modules (see structure above) | Architect | High | 4 hr |
| 4.3 | Define `__all__` in `__init__.py` with clean public API | Architect | Medium | 1 hr |
| 4.4 | Extract hardcoded values to `config.py` constants | Developer | Medium | 2 hr |
| 4.5 | Decompose `fit()` method into smaller private methods | Developer | Medium | 4 hr |
| 4.6 | Eliminate 4-branch DataFrame construction with a builder pattern | Developer | Medium | 2 hr |
| 4.7 | Make `fit()` return type consistent (always tuple, use empty DF for predictions) | Architect | High | 2 hr |

### Phase 5: Dependency & Build Modernization (Week 5-6)

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 5.1 | Migrate to `pyproject.toml` (PEP 621) | Architect | Medium | 3 hr |
| 5.2 | Make `xgboost`, `lightgbm`, `mlflow` optional extras | Architect | High | 2 hr |
| 5.3 | Pin minimum versions for all dependencies | Security | High | 1 hr |
| 5.4 | Add `requirements.lock` or adopt poetry/uv lock file | Security | Medium | 1 hr |
| 5.5 | Single-source version (remove from setup.py, keep in pyproject.toml) | Architect | Medium | 1 hr |
| 5.6 | Remove or implement CLI properly | Architect | Low | 2 hr |

### Phase 6: Testing & CI Hardening (Week 6-8)

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 6.1 | Add negative tests (invalid inputs, edge cases) | Architect | High | 4 hr |
| 6.2 | Add coverage reporting with minimum threshold (80%+) | Architect | Medium | 1 hr |
| 6.3 | Add mypy to CI pipeline | Developer | Medium | 1 hr |
| 6.4 | Test Windows timeout behavior | Security | Medium | 2 hr |
| 6.5 | Add thread-safety tests for concurrent instances | Security | Low | 3 hr |
| 6.6 | Isolate MLflow tests with proper fixtures (no env var dependency) | Architect | Medium | 2 hr |
| 6.7 | Add integration tests for optional dependency scenarios | Architect | Medium | 2 hr |

### Phase 7: Advanced Improvements (Week 8+)

| # | Task | Perspective | Severity | Effort |
|---|------|-------------|----------|--------|
| 7.1 | Make CLASSIFIERS/REGRESSORS lists immutable (tuple or frozenset) | Security | Low | 30 min |
| 7.2 | Add `max_models` parameter to limit resource usage | Security | Medium | 1 hr |
| 7.3 | Add `n_jobs` parameter (don't hardcode `-1`) | Security | Medium | 30 min |
| 7.4 | Add `memory_limit` parameter or dataset size warnings | Security | Low | 2 hr |
| 7.5 | Support `fit(X, y)` API without requiring test data | Architect | High | 4 hr |
| 7.6 | Add progress callback API (replace tqdm coupling) | Architect | Low | 2 hr |
| 7.7 | Add model serialization support (save/load trained models) | Architect | Medium | 4 hr |

---

## 5. Implementation Roadmap

```
Week 1-2:  Phase 1 (Critical Fixes)          ← Patch release v0.2.17
Week 2-3:  Phase 2 (Logging & Errors)         ← Part of v0.3.0
Week 3-4:  Phase 3 (Type Hints & Docs)        ← Part of v0.3.0
Week 4-6:  Phase 4 (Architecture Refactor)    ← v0.3.0 release
Week 5-6:  Phase 5 (Build Modernization)      ← v0.3.0 release
Week 6-8:  Phase 6 (Testing & CI)             ← v0.3.1
Week 8+:   Phase 7 (Advanced)                 ← v0.4.0
```

### Release Strategy

- **v0.2.17** (Patch): Critical fixes only — no API changes, backward compatible
- **v0.3.0** (Minor): Refactored internals, logging, type hints, optional deps — some deprecation warnings for old patterns
- **v0.4.0** (Minor): New `fit(X, y)` API, model serialization, advanced resource controls

### Backward Compatibility Notes

The following changes in v0.3.0 may affect users:
1. `warnings.filterwarnings("ignore")` removed — users may see sklearn warnings they didn't before
2. `pd.set_option` no longer called — DataFrame display may change
3. `xgboost`/`lightgbm` become optional — users need `pip install lazypredict[boost]`
4. `fit()` will always return a tuple (scores, predictions) — predictions is empty DF when `predictions=False`
5. Module path `lazypredict.Supervised` preserved via re-exports for compatibility

---

## Issue Count Summary

| Perspective | Critical | High | Medium | Low | Total |
|-------------|----------|------|--------|-----|-------|
| Architect   | 1 | 8 | 7 | 3 | 19 |
| Developer   | 1 | 5 | 6 | 6 | 18 |
| Security    | 1 | 4 | 4 | 1 | 10 |
| **Total**   | **3** | **17** | **17** | **10** | **47** |
