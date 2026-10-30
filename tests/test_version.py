"""The package version must be declared consistently everywhere it appears.

A GitHub release tagged ``v0.3.1`` was once created on a commit that still
said ``0.3.0``, so the publish workflow rebuilt 0.3.0 and PyPI rejected it.
These checks keep every version string in sync with ``lazypredict.__version__``.
"""

import pathlib
import re
import sys

import pytest

import lazypredict

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", lazypredict.__version__)


def test_pyproject_reads_version_from_package():
    pyproject = _read("pyproject.toml")
    assert re.search(r'^dynamic = \["version"\]', pyproject, re.MULTILINE)
    assert 'version = {attr = "lazypredict.__version__"}' in pyproject


@pytest.mark.parametrize(
    "relative_path, pattern",
    [
        ("recipe/meta.yaml", r'\{% set version = "([^"]+)" %\}'),
        ("CITATION.cff", r"^version: (\S+)$"),
        (".bumpversion.cfg", r"^current_version = (\S+)$"),
    ],
)
def test_version_strings_match_package(relative_path, pattern):
    if not (ROOT / relative_path).exists():
        pytest.skip(f"{relative_path} is not part of the source distribution")
    match = re.search(pattern, _read(relative_path), re.MULTILINE)
    assert match, f"no version found in {relative_path}"
    assert match.group(1) == lazypredict.__version__, relative_path


def test_release_tag_check_script():
    if not (ROOT / ".github" / "scripts" / "check_release_version.py").exists():
        pytest.skip("the release scripts are not part of the source distribution")
    sys.path.insert(0, str(ROOT / ".github" / "scripts"))
    try:
        import check_release_version
    finally:
        sys.path.pop(0)
    version = lazypredict.__version__
    assert check_release_version.main(["prog", f"v{version}"]) == 0
    assert check_release_version.main(["prog", version]) == 0
    assert check_release_version.main(["prog", "v999.0.0"]) == 1
