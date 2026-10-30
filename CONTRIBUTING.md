---
title: Contributing
---

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

# Types of Contributions

## Report Bugs

Report bugs at <https://github.com/shankarpandala/lazypredict/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Your Python version and virtual environment details.
-   Any details about your local setup that might be helpful in
    troubleshooting.
-   Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

## Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to
implement it.

## Write Documentation

Lazy Predict could always use more documentation, whether as part of the
official Lazy Predict docs, in docstrings, or even on the web in blog
posts, articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/shankarpandala/lazypredict/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to
    implement.
-   Remember that this is a volunteer-driven project, and that
    contributions are welcome.

# Get Started!

Ready to contribute? Here's how to set up `lazypredict` for
local development.

1.  Fork the `lazypredict` repo on GitHub.

2.  Clone your fork locally:

    ```shell
    git clone git@github.com:your_name_here/lazypredict.git
    ```

3.  Install your local copy into a virtual environment:

    ```shell
    cd lazypredict/
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -e ".[dev]"
    ```

4.  Create a branch for local development:

    ```shell
    git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

5.  When you're done making changes, check that your changes pass
    flake8 and the tests:

    ```shell
    flake8 lazypredict tests
    pytest
    ```

    You can also run the test suite on every supported Python version with tox:

    ```shell
    pip install tox
    tox
    ```

6.  Commit your changes and push your branch to GitHub:

    ```shell
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

7.  Submit a pull request through the GitHub website.

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.md.
3.  The pull request should work for Python 3.10 through 3.14. GitHub
    Actions CI will automatically run the tests for all supported
    Python versions.

# Tips

To run a subset of tests:

```shell
pytest tests/test_supervised.py
```

To run tests with coverage:

```shell
pytest --cov=lazypredict --cov-report=term-missing
```

# Releasing

Releases are cut from `dev` and published to PyPI by
`.github/workflows/publish.yml`, which runs when a GitHub release is
**published** (pushing a tag on its own does not publish anything).

1.  Bump the version. `lazypredict/__init__.py` is the single source of
    truth (`pyproject.toml` reads it); `recipe/meta.yaml`, `CITATION.cff`
    and `.bumpversion.cfg` must carry the same string, which
    `tests/test_version.py` enforces:

    ```shell
    pip install bump-my-version
    bump-my-version bump minor        # or: patch / major
    ```

    For a pre-release, pass the exact version instead, e.g.
    `bump-my-version bump --new-version 0.5.0a1`; `bump-my-version bump pre_n`
    then goes to `0.5.0a2`, and `--new-version 0.5.0` finalizes it.

2.  Move the "Unreleased" notes in `HISTORY.md` under the new version and
    update `date-released` in `CITATION.cff`.

3.  Open a pull request into `dev`, wait for CI to pass, and merge it.

4.  On GitHub, draft a new release **on the merged commit** with tag
    `v<version>` (for example `v0.4.0`; tick "Set as a pre-release" for
    `a`/`b`/`rc` versions) and publish it.

The publish workflow first checks that the tag matches
`lazypredict.__version__` and fails before building anything if it does
not, then runs the tests, builds and smoke-tests the sdist and wheel,
uploads them to PyPI via Trusted Publishing and attaches them to the
GitHub release. If an upload fails for a transient reason, re-run it from
the Actions tab (`Publish to PyPI` > `Run workflow`, entering the tag).

Instead of step 4 you can push an annotated tag on the merged commit and run
the workflow manually for it:

```shell
git tag -a v0.4.0 -m "0.4.0" && git push origin v0.4.0
gh workflow run publish.yml --ref v0.4.0 -f tag=v0.4.0
```

The workflow then creates the GitHub release, with generated notes, after
the upload to PyPI has succeeded.

The conda-forge feedstock picks up new PyPI releases automatically through
the conda-forge autotick bot; `recipe/meta.yaml` is only used by CI to check
that the package still builds with conda.
