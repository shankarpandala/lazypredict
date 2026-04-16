---
title: 'LazyPredict: Rapid Multi-Model Benchmarking for Machine Learning and Time Series Forecasting in Python'
tags:
  - Python
  - machine learning
  - automated machine learning
  - model selection
  - time series forecasting
  - scikit-learn
authors:
  - name: Shankar Rao Pandala
    orcid: 0009-0006-2234-6944
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 April 2026
bibliography: paper.bib
---

# Summary

LazyPredict is a Python library for rapid, zero-configuration benchmarking of
machine learning models. Given a training and test set, it automatically
discovers, fits, and evaluates all applicable estimators from scikit-learn
[@pedregosa2011] and popular gradient boosting libraries, returning a ranked
comparison table of performance metrics in a single function call. The library
supports three tasks: classification (over 30 classifiers reporting accuracy,
balanced accuracy, ROC AUC, and F1 score), regression (over 40 regressors
reporting adjusted R-squared, R-squared, and RMSE), and time series forecasting
(26+ models reporting MAE, RMSE, MAPE, SMAPE, MASE, and R-squared). LazyPredict
is designed for the *survey* phase of model selection --- identifying which model
families merit further investigation --- rather than for hyperparameter
optimization, making it complementary to full AutoML frameworks.

# Statement of Need

Model selection is a fundamental step in applied machine learning, yet it is
often performed in an ad hoc manner. Researchers typically instantiate a
handful of familiar algorithms, fit them individually, and compare results
manually. This workflow is tedious, error-prone, and biased toward algorithms
the practitioner already knows, potentially overlooking better-suited model
families.

A growing ecosystem of AutoML frameworks addresses this problem through
automated search over both model types and hyperparameter configurations.
Auto-sklearn [@feurer2015] uses Bayesian optimization and meta-learning over
scikit-learn pipelines. TPOT [@olson2016] applies genetic programming to evolve
optimal pipelines but is no longer actively maintained. PyCaret [@pycaret2020]
offers a low-code interface for end-to-end machine learning workflows covering
classification, regression, clustering, and time series forecasting. FLAML
[@wu2021] from Microsoft Research employs cost-frugal optimization to find
strong models under resource constraints. AutoGluon [@autogluon2020] from AWS
combines multi-layer model ensembling with deep learning integration across
tabular, text, image, and time series tasks.

These frameworks optimize for *finding the best model*, a computationally
expensive process that can require hours of search time. However, many
scientific workflows require a preliminary step: quickly surveying which
algorithm families perform well on a given dataset before committing to an
expensive tuning pipeline. LazyPredict is designed specifically for this
survey step. With two lines of code, users obtain a comprehensive comparison
of all applicable models using default hyperparameters, completing in seconds
to minutes rather than hours:

```python
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

This design serves several use cases in scientific computing:

- **Exploratory model selection**: Rapidly narrowing the space of candidate
  algorithms on a new dataset before investing compute in hyperparameter tuning.
- **Reproducible baselines**: Generating a standardized set of default-parameter
  results that serve as a reference point for evaluating tuned models.
- **Pedagogy**: Demonstrating the diversity of machine learning algorithms and
  their relative strengths across different data characteristics, enabling
  students to build intuition about algorithm behavior.
- **Time series model surveys**: Benchmarking statistical methods (ETS, ARIMA,
  Theta), machine learning regressors with engineered lag features, deep
  learning architectures (LSTM, GRU), and pretrained foundation models
  (TimesFM; @das2024) in a single call.

# State of the Field

\autoref{tab:comparison} summarizes how LazyPredict relates to established
AutoML frameworks.

: Comparison of LazyPredict with established AutoML frameworks. \label{tab:comparison}

| Capability | LazyPredict | Auto-sklearn | TPOT | PyCaret | FLAML | AutoGluon |
|:-----------|:-----------:|:------------:|:----:|:-------:|:-----:|:---------:|
| Zero-config survey | Yes | No | No | Partial | No | No |
| Setup complexity | 2 lines | 10+ lines | 10+ lines | 5+ lines | 10+ lines | 3+ lines |
| Hyperparameter tuning | Optional | Core | Core | Core | Core | Core |
| Time series forecasting | 26+ models | No | No | ~30 models | Partial | Yes |
| GPU acceleration | Yes | No | No | Partial | No | Yes |
| Core dependencies | 6 packages | 15+ packages | 10+ packages | 30+ packages | 8+ packages | 20+ packages |
| Actively maintained | Yes | Inactive | Inactive | Yes | Yes | Yes |

Auto-sklearn, while foundational to the AutoML field, has been inactive since
2023 with no new PyPI releases, and TPOT is similarly unmaintained as its
developers work on a successor [@olson2016]. PyCaret and AutoGluon remain
actively developed and offer the most comprehensive feature sets, but their
primary goal --- automated pipeline optimization --- differs from LazyPredict's
focus on rapid survey. FLAML occupies a middle ground with cost-frugal search,
but still requires configuration and targets model optimization rather than
broad comparison.

LazyPredict's differentiator is *deliberate simplicity*: it evaluates every
model at default parameters with no search overhead, giving researchers an
immediate landscape view. This makes it a natural first step before using any
of the above frameworks for focused optimization.

# Software Design

LazyPredict comprises approximately 6,800 lines of source code organized into
a modular architecture anchored by a Template Method pattern.

**Class hierarchy.** The abstract base class `LazyEstimator` defines the
benchmarking workflow: input validation, automatic preprocessing, model
iteration with progress tracking, metric computation, and results assembly.
Three concrete subclasses implement task-specific logic:

- `LazyClassifier`: dynamically discovers classifiers via scikit-learn's
  `all_estimators()`, appends optional XGBoost [@chen2016], LightGBM [@ke2017],
  and CatBoost [@prokhorenkova2018] classifiers, and reports classification
  metrics.
- `LazyRegressor`: follows the same pattern for regression estimators with
  adjusted R-squared, R-squared, and RMSE.
- `LazyForecaster`: benchmarks time series models organized into five
  categories --- baselines (Naive, SeasonalNaive), statistical models
  (ExponentialSmoothing, SARIMAX, AutoARIMA via statsmodels [@seabold2010] and
  pmdarima [@smith2017]), ML regressors using engineered lag features, deep
  learning models (LSTM, GRU via PyTorch; @paszke2019), and the TimesFM
  foundation model [@das2024].

**Automatic preprocessing.** Each model is wrapped in a scikit-learn `Pipeline`
with a `ColumnTransformer` that automatically detects numeric and categorical
features. Numeric features receive mean imputation and standard scaling.
Categorical features are split by cardinality: low-cardinality columns
(configurable threshold, default 11) use one-hot or ordinal encoding, while
high-cardinality columns default to ordinal encoding to prevent feature
explosion. Users can select from four encoding strategies: one-hot, ordinal,
target encoding, or binary encoding.

**Time series feature engineering.** For ML-based forecasters, `LazyForecaster`
constructs lag features ($y_{t-1}, \ldots, y_{t-k}$), rolling statistics (mean
and standard deviation over configurable windows), and first-difference features.
Seasonal period detection uses the autocorrelation function (ACF), selecting the
first peak exceeding the $2/\sqrt{n}$ significance threshold. Cross-validation
uses `TimeSeriesSplit` with expanding windows to respect temporal ordering.

**Fault tolerance.** Models that fail to converge, raise exceptions, or exceed
a per-model wall-clock timeout are logged and skipped without interrupting the
benchmark. This graceful degradation is essential for surveying heterogeneous
model families, where some algorithms may be incompatible with specific data
characteristics.

**GPU acceleration.** When `use_gpu=True`, LazyPredict applies backend-specific
parameters (XGBoost: `device="cuda"`, LightGBM: `device="gpu"`, CatBoost:
`task_type="GPU"`) and discovers cuML (RAPIDS) GPU-native estimators at runtime.
The library falls back to CPU transparently if no CUDA device is available.

**Optional extensions.** Additional modules provide hyperparameter tuning via
Optuna [@akiba2019] and scikit-learn search, model explainability through
permutation importance and SHAP [@lundberg2017], weighted and stacked ensemble
construction, MLflow experiment tracking, and PySpark/Dask DataFrame
auto-conversion.

# Research Impact Statement

LazyPredict was first released on PyPI in November 2019 and has been publicly
available for over six years. The library has accumulated over 3,200 GitHub
stars, approximately 6,000 weekly PyPI downloads, and over 45 citations in
Google Scholar across diverse research domains. It is also available on
conda-forge and supports Python 3.9 through 3.14.

Peer-reviewed publications using LazyPredict span multiple fields. Mahgoub
[@mahgoub2023] used LazyPredict for rapid model comparison in a heart failure
prediction study. Fellah et al. [@fellah2024] applied it to benchmark
classifiers for Word2Vec-based spam detection. Al-Hashimi and Alyaarubi
[@alhashimi2023] employed it for initial model selection in sepsis disease
prediction before applying hyperparameter tuning. Varzandeh et al.
[@varzandeh2025] used it as part of a systematic ML approach for modeling the
thermal conductivity of methane. Olugbade et al. [@olugbade2024] used it to
compare classifiers for predicting PIC50 values of COVID-19 compounds. These
citations demonstrate the library's utility as a practical tool for
establishing baselines and narrowing model search spaces across scientific
disciplines.

The library has also been adopted in educational settings, featured in courses
and tutorials on platforms including Kaggle, Towards Data Science, GeeksforGeeks,
and Analytics Vidhya, as well as in video tutorials by data science educators.

# AI Usage Disclosure

Generative AI tools (GitHub Copilot and Claude) were used during development to
assist with code generation, test writing, and documentation drafting. All
AI-generated content was reviewed, tested, and validated by the author prior to
inclusion. This paper was drafted with AI assistance and reviewed and edited by
the author.

# Acknowledgements

The author thanks the contributors to the LazyPredict project, including Breno
Batista da Silva, Murat Toprak, Chanin Nantasenamat, Felipe Sassi, Shyambhu
Mukherjee, Johannes Schoeck, and all community members who have submitted
issues and pull requests. The author also acknowledges the developers of
scikit-learn, pandas [@mckinney2010], NumPy [@harris2020], and the broader
Python scientific computing ecosystem on which LazyPredict depends.

# References
