"""Supervised tuning robustness and results (issues #604, #605)."""

import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression, Ridge

import lazypredict.tuning as tuning
from lazypredict.search_spaces import SEARCH_SPACES
from lazypredict.Supervised import LazyClassifier, LazyRegressor

pytest.importorskip("optuna")


def _split(loader, n_train, as_frame=False):
    X, y = loader(return_X_y=True, as_frame=as_frame)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# -- #604: tuning never discards the benchmark --------------------------------

def test_cv_regressors_without_search_space_do_not_crash():
    reg = LazyRegressor(regressors=["ElasticNetCV", "LassoCV", "Ridge"], tune=True,
                        tune_top_k=3, tune_trials=2, n_jobs=1)
    scores, _ = reg.fit(*_split(load_diabetes, 300))
    assert len(scores) == 3
    status = reg.tuned_scores_["Status"]
    assert status["ElasticNetCV"] == "no search space"
    assert status["Ridge"] == "tuned"


def test_failing_model_is_reported_not_raised(monkeypatch):
    original = tuning.tune_supervised_optuna

    def flaky(model_name, *args, **kwargs):
        if model_name == "DecisionTreeClassifier":
            raise RuntimeError("boom")
        return original(model_name, *args, **kwargs)

    monkeypatch.setattr(tuning, "tune_supervised_optuna", flaky)
    clf = LazyClassifier(classifiers=["DecisionTreeClassifier", "KNeighborsClassifier"],
                         tune=True, tune_top_k=2, tune_trials=2, n_jobs=1)
    clf.fit(*_split(load_breast_cancer, 400))
    assert clf.tuned_scores_.loc["DecisionTreeClassifier", "Status"] == "failed: boom"
    assert clf.tuned_scores_.loc["KNeighborsClassifier", "Status"] == "tuned"


@pytest.mark.parametrize("option", ["tune_top_k", "tune_trials"])
def test_tune_options_are_validated(option):
    with pytest.raises(ValueError, match=option):
        LazyClassifier(**{option: 0})


def test_sklearn_backend_survives_small_folds_and_covers_all_spaces():
    clf = LazyClassifier(classifiers=["KNeighborsClassifier", "ExtraTreesClassifier"], tune=True,
                         tune_backend="sklearn", tune_top_k=2, tune_trials=6, n_jobs=1)
    clf.fit(*_split(load_iris, 50))  # 40-row folds: some n_neighbors candidates are invalid
    assert set(clf.tuned_scores_["Status"]) == {"tuned"}


def test_every_search_space_converts_to_sklearn_distributions():
    for name in SEARCH_SPACES:
        dists = tuning._optuna_to_sklearn_dist(name)
        assert dists, name
        assert all(isinstance(v, list) or hasattr(v, "rvs") for v in dists.values()), name


# -- #605: results are usable and reproducible ---------------------------------

def test_best_params_are_dicts_and_applied_to_tuned_models():
    clf = LazyClassifier(classifiers=["RandomForestClassifier"], tune=True,
                         tune_top_k=1, tune_trials=3, n_jobs=1)
    X_train, X_test, y_train, y_test = _split(load_breast_cancer, 400, as_frame=True)
    clf.fit(X_train, X_test, y_train, y_test)
    params = clf.tuned_scores_.loc["RandomForestClassifier", "Best Params"]
    assert isinstance(params, dict) and params == clf.tuned_params_["RandomForestClassifier"]
    tuned = clf.tuned_models_["RandomForestClassifier"]
    fitted = tuned.named_steps["classifier"].get_params()
    assert all(fitted[key] == value for key, value in params.items())
    assert len(tuned.predict(X_test)) == len(X_test)


def test_tuning_is_reproducible():
    def run():
        clf = LazyClassifier(classifiers=["KNeighborsClassifier"], tune=True,
                             tune_top_k=1, tune_trials=5, random_state=7, n_jobs=1)
        clf.fit(*_split(load_breast_cancer, 400))
        return clf.tuned_params_

    assert run() == run()


def test_search_space_fixes():
    assert "LassoCV" not in SEARCH_SPACES and "ElasticNetCV" not in SEARCH_SPACES
    assert not any(name.startswith("PassiveAggressive") for name in SEARCH_SPACES)
    assert "LogisticRegression" in SEARCH_SPACES
    lgbm = tuning._optuna_to_sklearn_dist("LGBMClassifier")
    assert lgbm["subsample_freq"] == [1]
    catboost = tuning._optuna_to_sklearn_dist("CatBoostClassifier")
    assert catboost["bootstrap_type"] == ["Bayesian"]


def test_flaml_backend():
    pytest.importorskip("flaml.automl")
    clf = LazyClassifier(classifiers=["RandomForestClassifier", "SVC"], tune=True,
                         tune_backend="flaml", tune_top_k=2, tune_trials=3, tune_timeout=30, n_jobs=1)
    clf.fit(*_split(load_breast_cancer, 400))
    status = clf.tuned_scores_["Status"]
    assert status["SVC"] == "no search space"  # FLAML has no SVC learner mapping
    assert status["RandomForestClassifier"] == "tuned"
    assert 0 < clf.tuned_scores_.loc["RandomForestClassifier", "Best Score"] <= 1


def test_flaml_defaults_as_best_config_count_as_tuned():
    # Issue #622: an empty "Best Params" (FLAML's best config equals the
    # defaults) was reported as "failed: no successful trial".
    pytest.importorskip("flaml.automl")
    clf = LazyClassifier(classifiers=["LogisticRegression"], tune=True, tune_backend="flaml",
                         tune_top_k=1, tune_trials=1, tune_timeout=30, n_jobs=1)
    clf.fit(*_split(load_breast_cancer, 400))
    row = clf.tuned_scores_.loc["LogisticRegression"]
    assert row["Status"] == "tuned"
    assert 0 < row["Best Score"] <= 1
    assert "LogisticRegression" in clf.tuned_models_


def test_flaml_task_detection_covers_catboost_style_classifiers():
    from lazypredict.tuning import _is_classifier_class

    class CatBoostLikeClassifier:  # does not subclass ClassifierMixin
        _estimator_type = "classifier"

    assert _is_classifier_class(CatBoostLikeClassifier)
    assert _is_classifier_class(LogisticRegression)
    assert not _is_classifier_class(Ridge)


def test_tuned_models_predict_and_explain_numpy_input():
    # Issue #631: tuned pipelines rejected NumPy input and predict() could not use them.
    X, y = load_breast_cancer(return_X_y=True)
    clf = LazyClassifier(classifiers=["LogisticRegression"], tune=True, tune_top_k=1, tune_trials=3)
    clf.fit(X[:400], X[400:], y[:400], y[400:])
    tuned = clf.predict(X[400:], model_name="LogisticRegression", tuned=True)
    assert tuned.shape == (len(X) - 400,)
    importance = clf.explain(X[400:], y[400:], n_repeats=2, tuned=True)
    assert list(importance.columns) == ["LogisticRegression"]
    untuned = LazyClassifier(classifiers=["LogisticRegression"])
    untuned.fit(X[:400], X[400:], y[:400], y[400:])
    with pytest.raises(ValueError, match="No tuned models"):
        untuned.predict(X[400:], tuned=True)

