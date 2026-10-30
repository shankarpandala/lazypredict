"""A broken optional dependency must not break ``import lazypredict`` (issue #611)."""

import os
import pathlib
import subprocess
import sys
import textwrap
import warnings

import pytest

from lazypredict._optional import warn_if_broken

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _broken_package(root, name, statement):
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_text(statement + "\n")


def test_broken_boosters_are_skipped_with_a_warning(tmp_path):
    # What XGBoost and LightGBM raise on macOS when libomp is missing.
    _broken_package(
        tmp_path, "xgboost", 'raise ValueError("libxgboost.dylib could not be loaded")'
    )
    _broken_package(
        tmp_path, "lightgbm", 'raise OSError("Library not loaded: @rpath/libomp.dylib")'
    )
    script = textwrap.dedent(
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from lazypredict import Supervised, TimeSeriesForecasting
        for module in (Supervised, TimeSeriesForecasting):
            assert not module._XGBOOST_AVAILABLE
            assert not module._LIGHTGBM_AVAILABLE
        names = {name for name, _ in Supervised.CLASSIFIERS + Supervised.REGRESSORS}
        assert not names & {"XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor"}
        assert "RandomForestClassifier" in names
        messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("xgboost is installed" in m and "libxgboost" in m for m in messages), messages
        assert any("lightgbm is installed" in m and "libomp" in m for m in messages), messages
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO_ROOT)])}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_missing_package_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_broken("somepackage", ModuleNotFoundError("No module named 'somepackage'"))


def test_broken_package_warns():
    with pytest.warns(RuntimeWarning, match=r"pmdarima is installed but could not be imported"):
        warn_if_broken("pmdarima", ValueError("numpy.dtype size changed"))
