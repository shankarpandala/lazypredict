# LazyPredict Roadmap — v0.3.x and Beyond

**Last Updated:** 2026-03-10
**Current Version:** 0.3.0a5

---

## Vision

Extend LazyPredict from a "run all models with defaults" tool into a lightweight
AutoML-lite toolkit that covers **model comparison → explainability → tuning →
forecasting optimization** — while keeping the "3 lines of code" philosophy.

---

## Summary Table

| Phase | Feature | Dep | Lines of Code | Priority |
|-------|---------|-----|---------------|----------|
| **0.3.1** | `explain()` with `permutation_importance` | None | ~100 | P0 |
| **0.3.1** | Auto-convert PySpark/Dask DataFrames | None | ~20 | P0 |
| **0.3.2** | `explain(method="shap")` with auto-explainer | `shap` | ~200 | P1 |
| **0.3.2** | `tune(top_k=5)` with Optuna | `optuna` | ~300 | P1 |
| **0.3.3** | EBM model from InterpretML | `interpret` | ~30 | P2 |
| **0.3.3** | Dask joblib backend for CV | `dask` | ~15 | P2 |
| **0.3.3** | Forecasting hyperparameter tuning | `optuna` | ~400 | P1 |
| **0.3.3** | Multi-step horizon optimization | None | ~200 | P1 |
| **0.3.3** | Forecast loss function selection | None | ~30 | P2 |
| **0.3.3** | Ensemble top-k forecasters | None | ~150 | P2 |
| **0.3.3** | Seasonal period as tunable hyperparameter | None | ~50 | P2 |
| **0.4.0** | Spark MLlib models (`LazySparkClassifier`) | `pyspark` | ~500+ | P3 |
| **0.4.0** | FLAML as optional tune backend | `flaml` | ~100 | P3 |

---

## Phase Overview

| Phase | Version | Theme | Status |
|-------|---------|-------|--------|
| 0 | 0.3.0a3 | Core refactor, time series, GPU | Done |
| 1 | 0.3.0a4 | Explainability (zero-dep) + data auto-convert | Done |
| 2 | 0.3.0a4 | SHAP explainability + supervised HPO | Done |
| 3 | 0.3.0a4 | Forecasting optimization + EBM + Dask CV | Done |
| 4 | 0.3.0a4 | Spark MLlib + FLAML backend | Done |

---

## Phase 1 — v0.3.1: Explainability & Data Auto-Convert

**Goal:** Let users understand *why* models rank the way they do, and accept
PySpark/Dask DataFrames without manual conversion. Zero new dependencies.

### 1.1 `explain()` method with `permutation_importance` (P0, ~100 LOC)

- **Scope:** Add `explain()` to `LazyEstimator` base class.
- **Default method:** `sklearn.inspection.permutation_importance` — works for
  ALL model types (tree, linear, SVM, KNN), requires zero extra dependencies,
  and operates on the full Pipeline (no need to extract inner estimator).
- **Returns:** `pd.DataFrame` with feature importance rankings per model.
- **Dependencies:** None (sklearn built-in).

### 1.2 Auto-convert PySpark/Dask DataFrames (P0, ~20 LOC)

- **Scope:** In `_validate_fit_inputs()`, detect `pyspark.sql.DataFrame` and
  `dask.dataframe.DataFrame` and auto-call `.toPandas()` / `.compute()` with
  a logged warning.
- **Dependencies:** None (optional imports only).

---

## Phase 2 — v0.3.2: SHAP Explainability & Supervised HPO

**Goal:** Add SHAP-based explainability and opt-in "top-k then tune" HPO for
classification and regression.

### 2.1 SHAP-based Explainability (P1, ~200 LOC)

- `explain(method="shap")` with auto-explainer selection:
  - `TreeExplainer` for tree/boosting models
  - `LinearExplainer` for linear models
  - `permutation_importance` as fallback for all others
- Extract model via `pipe.named_steps["regressor"]` and pre-transform data.
- SHAP 0.50+ requires Python >= 3.11 — use version-conditional dep pinning.
- **Dependencies:** `shap` (new optional extra: `pip install lazypredict[explain]`).

### 2.2 Optuna-based tuning for `LazyClassifier` / `LazyRegressor` (P1, ~300 LOC)

- **Architecture:** Two-phase workflow:
  1. **Phase A (existing):** Run all models with defaults. Rank by metric.
  2. **Phase B (new, opt-in):** Tune the top-k models from Phase A using Optuna.

- **API surface:**
  ```python
  LazyClassifier(
      tune=False,             # Enable HPO (default off — stays "lazy")
      tune_top_k=5,           # Only tune top 5 models from Phase A
      tune_trials=50,         # Number of Optuna trials per model
      tune_timeout=60,        # Seconds per model tuning
      tune_backend="optuna",  # "optuna" (default), "sklearn"
  )
  ```

- **Search space registry:** New `search_spaces.py` module mapping model class
  names to Optuna search space definitions.
  - **Boosting:** learning_rate, n_estimators, max_depth, regularization
  - **Trees:** n_estimators, max_depth, min_samples_split/leaf
  - **Linear:** alpha, l1_ratio
  - **SVM:** C, kernel, gamma
  - **KNN:** n_neighbors, weights, metric
  - Models with no meaningful hyperparameters (e.g., DummyClassifier) are skipped.

- **Pruning:** Leverage `optuna-integration` callbacks for LightGBM, XGBoost,
  CatBoost to enable early stopping of unpromising trials.

- **Output:** Return both the default-ranking DataFrame AND a tuned-ranking
  DataFrame with best params per model.

### 2.3 sklearn fallback backend

- `tune_backend="sklearn"` uses `RandomizedSearchCV` — zero extra deps.
- Less sample-efficient than Optuna but available out of the box.

---

## Phase 3 — v0.3.3: Forecasting Optimization + EBM + Dask CV

**Goal:** Bring the "top-k then tune" HPO workflow to `LazyForecaster` with
forecasting-specific optimization techniques, add InterpretML EBM models,
and enable distributed CV via Dask.

### 3.1 Forecasting Hyperparameter Tuning (P1, ~400 LOC)

- **Extend `LazyForecaster` with `tune=True` support**, mirroring the Phase 2
  API but with forecasting-specific adaptations:

  ```python
  LazyForecaster(
      tune=False,             # Enable HPO (default off)
      tune_top_k=5,           # Tune top 5 forecasters from initial ranking
      tune_trials=30,         # Optuna trials per model
      tune_timeout=120,       # Seconds per model (forecasting is slower)
      tune_backend="optuna",  # "optuna" or "sklearn"
  )
  ```

- **Forecasting-specific search spaces** (`ts_search_spaces.py`):

  | Model Category | Hyperparameters to Tune |
  |---------------|------------------------|
  | **SARIMAX** | order (p,d,q), seasonal_order (P,D,Q,s), trend |
  | **Exponential Smoothing** | smoothing_level, smoothing_trend, smoothing_seasonal, damped_trend |
  | **AutoARIMA** | max_p, max_q, max_d, seasonal, stepwise, information_criterion |
  | **Theta** | theta parameter, deseasonalize |
  | **ML regressors** (Ridge_TS, RF_TS, etc.) | Same as supervised + n_lags, n_rolling window sizes |
  | **Boosting** (XGB_TS, LGBM_TS, CatBoost_TS) | learning_rate, n_estimators, max_depth + lag/rolling features |
  | **LSTM / GRU** | hidden_size, num_layers, dropout, learning_rate, batch_size, epochs |
  | **Naive / SeasonalNaive** | Skip (no meaningful hyperparams) |

- **Temporal cross-validation:** All tuning uses `TimeSeriesSplit` (expanding
  window) — never shuffle. Optuna's objective function wraps `TimeSeriesSplit`
  internally.

- **Feature engineering as hyperparameters:** `n_lags` and `n_rolling` window
  sizes are treated as tunable hyperparameters for ML/DL models, not just the
  model's own params. This is unique to the forecasting case.

- **Dependencies:** `optuna` (shared with Phase 2 extra).

### 3.2 Multi-step Horizon Optimization (P1, ~200 LOC)

- **Current state:** `LazyForecaster` forecasts `len(y_test)` steps ahead using
  recursive forecasting for ML models.
- **Enhancement:** Add `horizon_strategy` parameter:
  - `"recursive"` (default, current behavior): Train one model, predict one
    step, feed prediction back as a lag feature, repeat.
  - `"direct"`: Train separate models for each forecast step. More accurate for
    long horizons but more expensive.
  - `"multi_output"`: Use sklearn's `MultiOutputRegressor` to predict all steps
    at once from a single model.
- **Tuning integration:** When `tune=True`, include `horizon_strategy` as a
  categorical hyperparameter for ML models (let Optuna find the best strategy).

### 3.3 Forecast Loss Function Selection (P2, ~30 LOC)

- Allow users to specify which metric to optimize during tuning via
  `tune_metric`:
  ```python
  LazyForecaster(
      tune=True,
      tune_metric="SMAPE",   # Optimize SMAPE during HPO (default: RMSE)
  )
  ```
- **Why it matters:** RMSE penalizes outliers heavily; MAE is robust; MAPE is
  undefined at zero; SMAPE is bounded. The best metric depends on the use case
  (e.g., demand forecasting often uses SMAPE or MASE).

### 3.4 Ensemble Top-K Forecasters (P2, ~150 LOC)

- After tuning, optionally ensemble the top-k tuned models using:
  - **Simple average** of predictions
  - **Inverse-error weighted average** (weight inversely proportional to
    validation error)
  - **Stacking** (train a meta-learner on top-k model predictions)
- **API:**
  ```python
  forecaster = LazyForecaster(tune=True, tune_top_k=5)
  scores, preds = forecaster.fit(y_train, y_test)
  ensemble_pred = forecaster.ensemble(method="weighted_average")
  ```
- **Motivation:** AutoGluon's success validates that ensembling diverse models
  with reasonable defaults often beats extensively tuning a single model.

### 3.5 Seasonal Period as Tunable Hyperparameter (P2, ~50 LOC)

- **Current state:** `seasonal_period` is either user-specified or
  auto-detected via ACF.
- **Enhancement:** When `tune=True`, allow Optuna to search over plausible
  seasonal periods (e.g., `[7, 12, 24, 52, 365]`) for statistical models
  like HoltWinters and SARIMAX, rather than relying on a single ACF estimate.

### 3.6 InterpretML EBM Model (P2, ~30 LOC)

- Add `ExplainableBoostingClassifier` / `Regressor` from `interpret` as an
  optional model — inherently interpretable, competitive accuracy.
- **Dependencies:** `interpret` (optional extra).

### 3.7 Dask Joblib Backend for Cross-Validation (P2, ~15 LOC)

- When Dask is available, register `dask.distributed` as the joblib backend
  for `cross_validate()` — distributes CV folds across workers.
- **Dependencies:** `dask` (optional, auto-detected).

---

## Phase 4 — v0.4.0: Advanced Integrations

### 4.1 Spark MLlib Models — `LazySparkClassifier` (P3, ~500+ LOC)

- Separate class that wraps Spark MLlib estimators (~12 model families).
- Only for users with existing Spark infrastructure and datasets >10GB.
- **Dependencies:** `pyspark` (optional extra).

### 4.2 FLAML as Optional Tune Backend (P3, ~100 LOC)

- `tune_backend="flaml"` uses FLAML's cost-frugal search algorithm.
- Leverages FLAML's battle-tested per-estimator search spaces.
- **Dependencies:** `flaml` (optional extra).

---

## Dependency Strategy

| Extra | Package | Phase | Reason |
|-------|---------|-------|--------|
| *(core)* | *(none new)* | 0.3.1 | Explainability + auto-convert |
| `explain` | `shap>=0.42` | 0.3.2 | SHAP explainability |
| `tune` | `optuna>=3.0` | 0.3.2, 0.3.3 | Primary HPO engine |
| `interpret` | `interpret>=0.4` | 0.3.3 | EBM models |
| `spark` | `pyspark>=3.3` | 0.4.0 | Spark MLlib integration |
| `flaml` | `flaml>=2.0` | 0.4.0 | Alternative HPO backend |

Phase 1 (v0.3.1) requires **zero new dependencies**.

---

## Design Principles

1. **Lazy by default.** HPO, explainability, and ensembling are all opt-in.
   `LazyClassifier().fit(X_train, X_test, y_train, y_test)` still works
   exactly as before with zero extra args.

2. **Top-k then tune.** Never waste compute tuning models that clearly don't
   fit the data. Run all models with defaults first, then tune the winners.

3. **Temporal integrity.** Forecasting optimization always respects time
   ordering — `TimeSeriesSplit`, no shuffling, expanding window CV.

4. **Forecasting-aware HPO.** Feature engineering params (`n_lags`,
   `n_rolling`, `seasonal_period`) are hyperparameters too, not just model
   knobs. This is what makes forecasting optimization different from
   supervised HPO.

5. **Minimal dependencies.** Each new capability is an optional extra.
   `pip install lazypredict` stays lightweight.

6. **Don't reinvent AutoML.** LazyPredict is a *comparison* tool, not a
   production pipeline builder. Keep it simple, keep it fast.

---

## What We Explicitly Won't Do

- **Hyperopt**: Unmaintained since Nov 2021. Optuna is strictly superior.
- **Auto-sklearn**: Inactive, promised refactor never shipped.
- **Ray Tune as default**: Too heavy for the typical laptop user. Available
  through FLAML if needed.
- **Single CASH study across 30+ models**: The combined search space is too
  large. Independent top-k tuning is faster and more practical.
- **Make HPO the default**: LazyPredict's value is speed. Tuning is opt-in.
