"""Fail fast when a release tag does not match the package version.

Used by the publish workflow so that a GitHub release whose tag disagrees
with ``lazypredict.__version__`` (e.g. tag ``v0.3.1`` on a commit still
versioned ``0.3.0``) stops before anything is built or uploaded to PyPI.

Usage::

    python .github/scripts/check_release_version.py v0.4.0
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def package_version() -> str:
    """Read ``__version__`` from ``lazypredict/__init__.py`` without importing it."""
    source = (ROOT / "lazypredict" / "__init__.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("Could not find __version__ in lazypredict/__init__.py")


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    tag = argv[1].strip()
    expected = tag[1:] if tag.startswith("v") else tag
    actual = package_version()
    if actual != expected:
        print(
            f"::error::Release tag '{tag}' does not match the package version "
            f"'{actual}'. Bump the version (see CONTRIBUTING.md, 'Releasing') "
            f"and re-create the release on the bumped commit."
        )
        return 1
    print(f"Release tag '{tag}' matches package version {actual}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
