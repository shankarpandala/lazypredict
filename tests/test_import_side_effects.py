"""Importing lazypredict must not patch scikit-learn (issue #591)."""

import os
import pathlib
import subprocess
import sys
import textwrap

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_import_does_not_call_patch_sklearn(tmp_path):
    # A fake ``sklearnex`` records whether patch_sklearn() gets called.
    fake = tmp_path / "sklearnex"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "CALLED = []\n"
        "def patch_sklearn(*args, **kwargs):\n"
        "    CALLED.append(True)\n"
    )
    script = textwrap.dedent(
        """
        import sklearnex
        import lazypredict
        from lazypredict.Supervised import INTEL_EXTENSION_AVAILABLE
        assert INTEL_EXTENSION_AVAILABLE is True
        assert sklearnex.CALLED == [], "lazypredict patched scikit-learn on import"
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO_ROOT)])}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
