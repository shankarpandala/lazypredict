# CI/CD Pipeline Audit & Automated Release Plan

**Audit Date:** 2026-02-28
**Current State:** Manual release via GitHub UI → publish.yml triggers
**Target State:** Fully automated release on merge to `master` via tagging strategy

---

## 1. Current Pipeline Diagnosis

### 1.1 Workflow Inventory

| File | Trigger | Purpose | Status |
|------|---------|---------|--------|
| `ci.yml` | push/PR to master,dev; release published | Lint + Test | **BROKEN** |
| `publish.yml` | release created (manual) | Build + Publish to PyPI | **PARTIALLY WORKING** |
| `docs.yml` | any PR | Build + deploy docs to gh-pages | **FRAGILE** |
| `codeql-analysis.yml` | push/PR to dev; weekly | Security scanning | **OUTDATED** |
| `update_citations.yml` | weekly cron; manual | Update README badge | OK |

### 1.2 Issues Found — `ci.yml`

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 1 | **`pytest \|\| true` — tests NEVER fail the build** | CRITICAL | Line 50: The `|| true` means pytest exit code is always 0. A release could ship with every test failing and CI would still be green. |
| 2 | **No test gate before publish** | CRITICAL | `publish.yml` runs independently — it does NOT wait for CI to pass. A broken release can be published. |
| 3 | **18-matrix build is wasteful** | HIGH | 3 OS × 6 Python versions = 18 jobs run on every push. Most issues are caught on 1 OS with 2-3 Python versions. Full matrix should only run on PRs to master. |
| 4 | **Flake8 second pass is exit-zero** | MEDIUM | Line 44: All style warnings are ignored. Only syntax errors caught. |
| 5 | **No caching** | MEDIUM | pip dependencies are re-downloaded on every run. Adds 1-2 min per job × 18 jobs = ~30 min wasted per push. |
| 6 | **No coverage reporting** | MEDIUM | No `pytest-cov`, no coverage thresholds, no badge. |

### 1.3 Issues Found — `publish.yml`

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 7 | **Requires manual GitHub Release creation** | HIGH | This is why you must publish manually. The workflow triggers on `release: types: [created]` which requires you to go to GitHub UI → Releases → Draft → Publish. |
| 8 | **Version rewriting with sed is fragile** | HIGH | Lines 33-35: Uses regex `sed` to rewrite version in `setup.py` and `__init__.py`. If the format changes slightly, sed silently produces wrong output. |
| 9 | **Version changes are NOT committed** | HIGH | The sed-modified files exist only in the build artifact. The repo still shows the old version after release. Version drift between PyPI and repo. |
| 10 | **Uses outdated actions** | MEDIUM | `actions/checkout@v2` and `actions/setup-python@v2` — should be v4/v5. |
| 11 | **No test artifact or sdist validation** | MEDIUM | `python -m build` output isn't validated with `twine check`. |
| 12 | **No TestPyPI stage** | MEDIUM | Publishes directly to production PyPI. No dry-run or staging. |
| 13 | **OIDC token but explicit repository-url** | LOW | Lines 8-9 configure OIDC (`id-token: write`) but line 44 sets `repository-url` explicitly. Should use Trusted Publishers via OIDC (no API tokens needed). |
| 14 | **No build artifact retention** | LOW | Built wheel/sdist not uploaded as GitHub artifact for debugging. |

### 1.4 Issues Found — `docs.yml`

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 15 | **Runs on EVERY PR, including drafts** | MEDIUM | Should only deploy docs on merge to master. |
| 16 | **Uses deprecated `ad-m/github-push-action@v0.6.0`** | MEDIUM | Very old action, unmaintained. |
| 17 | **Clones full repo again inside the workflow** | LOW | Line 33: `git clone ... --branch gh-pages` — redundant, could use peaceiris/actions-gh-pages. |
| 18 | **Pushes docs from PRs** | HIGH | Line 44: Docs are pushed to gh-pages on every PR, even unmerged ones. Should only deploy from master. |

### 1.5 Issues Found — `codeql-analysis.yml`

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 19 | **Uses `github/codeql-action/*@v1`** | CRITICAL | v1 was deprecated in December 2022 and removed. This workflow is **silently failing**. |
| 20 | **Only runs on `dev` branch** | MEDIUM | Should also scan `master`. |
| 21 | **Uses `actions/checkout@v2`** | LOW | Should be v4. |
| 22 | **Manual `git checkout HEAD^2`** | LOW | Line 27: This pattern was needed for old checkout versions. v4 handles PR head checkout natively. |

### 1.6 Issues Found — Version Management

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 23 | **Version in 4 places** | HIGH | `setup.py` (0.2.16), `__init__.py` (0.2.16), `.bumpversion.cfg` (0.2.16), `meta.yaml` (0.2.15 — STALE) |
| 24 | **`.bumpversion.cfg` misses `meta.yaml`** | MEDIUM | Only patches `setup.py` and `__init__.py`. |
| 25 | **No automated version bumping** | HIGH | Must manually run `bump2version` and create release. |
| 26 | **`meta.yaml` version is 0.2.15, not 0.2.16** | LOW | Already out of sync. |

### 1.7 Issues Found — Branch Strategy

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| 27 | **No branch protection rules documented** | HIGH | No evidence of required reviews, status checks, or merge restrictions. |
| 28 | **`dev` and `master` both accept direct pushes** | HIGH | CI triggers on push to both — implies no PR requirement. |
| 29 | **No release branching model** | HIGH | No `release/*` branches, no tags in CI triggers. |
| 30 | **`tox.ini` references Travis CI** | LOW | Line 8: `[travis]` section — Travis CI hasn't been used in years. |

---

## 2. Target Architecture

### 2.1 Branch Strategy (Trunk-Based with Release Tags)

```
Feature branches ──PR──→ dev ──PR──→ master ──tag──→ PyPI
                          │                      │
                      CI: lint+test          CI: full matrix
                      (fast, 3 jobs)         + build + publish
```

**Flow:**
1. Developer creates feature branch from `dev`
2. Opens PR to `dev` → fast CI runs (lint + test on ubuntu, Python 3.10/3.13)
3. Merges to `dev` → same fast CI
4. Opens PR from `dev` to `master` → full matrix CI (3 OS × 3 Python versions)
5. Merges to `master` → **automatically** creates a GitHub Release + tag based on version in source
6. Release created → `publish.yml` builds and publishes to PyPI
7. Docs deploy only on push to `master`

### 2.2 Automated Release Trigger Options

| Option | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **A. Tag-push triggers release** | Push `v*` tag → publish | Simple, explicit | Still requires manual `git tag` |
| **B. Version-diff detection** | Merge to master detects version change → auto-tag + release | Fully automatic | Slightly complex |
| **C. Conventional Commits + Release Please** | Google's release-please bot reads commit messages → auto-PR with changelog + version bump | Industry standard, changelogs | Requires conventional commit discipline |

**Recommendation: Option C (Release Please)** — it's the industry standard for open-source Python projects. It will:
- Auto-detect version bumps from commit messages (`feat:` → minor, `fix:` → patch, `feat!:` → major)
- Create a "Release PR" with updated changelog and version
- When that PR is merged, it creates the GitHub Release + tag automatically
- Then `publish.yml` picks up the release event and publishes to PyPI

---

## 3. New Workflow Designs

### 3.1 `ci.yml` — Redesigned

```yaml
name: CI

on:
  push:
    branches: [dev, master]
  pull_request:
    branches: [dev, master]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install flake8
      - name: Lint (syntax errors and undefined names)
        run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
      - name: Lint (style warnings)
        run: flake8 . --count --max-complexity=10 --statistics

  test-fast:
    # Runs on every push/PR — fast feedback
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.13']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -r requirements.txt pytest pytest-cov
      - name: Run tests with coverage
        run: pytest --cov=lazypredict --cov-report=xml --cov-fail-under=60
      - name: Upload coverage
        if: matrix.python-version == '3.12'
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  test-full:
    # Full matrix — only on PRs to master and release events
    if: github.base_ref == 'master' || github.ref == 'refs/heads/master'
    needs: lint
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.11', '3.13']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -r requirements.txt pytest
      - name: Run tests
        run: pytest
```

**Key changes:**
- Removed `|| true` — tests now actually gate the build
- Split into `lint` → `test-fast` (2 jobs, every push) and `test-full` (9 jobs, master PRs only)
- Added pip caching
- Added coverage with minimum threshold
- Added `concurrency` to cancel superseded runs
- Flake8 style warnings now fail the build (removed `--exit-zero`)

### 3.2 `release-please.yml` — NEW (Automated Releases)

```yaml
name: Release Please

on:
  push:
    branches: [master]

permissions:
  contents: write
  pull-requests: write

jobs:
  release-please:
    runs-on: ubuntu-latest
    outputs:
      release_created: ${{ steps.release.outputs.release_created }}
      tag_name: ${{ steps.release.outputs.tag_name }}
    steps:
      - uses: googleapis/release-please-action@v4
        id: release
        with:
          release-type: python
          package-name: lazypredict
          # Reads version from lazypredict/__init__.py
          # Creates a Release PR with changelog
          # On merge of Release PR, creates GitHub Release + tag
```

**What this does:**
1. Every push to `master` → release-please scans commit messages
2. If there are releasable changes (`feat:`, `fix:`, etc.) → creates/updates a "Release PR" with:
   - Bumped version in `__init__.py` and `setup.py`
   - Auto-generated `CHANGELOG.md`
3. When you merge the Release PR → creates GitHub Release + git tag
4. GitHub Release triggers `publish.yml` → package goes to PyPI

**You never need to visit the Releases page again.**

### 3.3 `publish.yml` — Redesigned

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read
  id-token: write  # OIDC for Trusted Publishers

jobs:
  # Gate: ensure CI passed for this ref
  ci-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: pip install -r requirements.txt pytest
      - name: Run tests before publish
        run: pytest

  build:
    needs: ci-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install build twine
      - name: Build package
        run: python -m build
      - name: Validate package
        run: twine check dist/*
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish-testpypi:
    needs: build
    runs-on: ubuntu-latest
    environment: testpypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: [build, publish-testpypi]
    if: github.event.release.prerelease == false
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

**Key changes:**
- Added `ci-check` gate — tests must pass before publish
- Removed fragile `sed` version rewriting (release-please handles version bumps in source)
- Added `twine check` for package validation
- Added TestPyPI staging step before production PyPI
- Uses OIDC Trusted Publishers (no API tokens needed — configure at pypi.org)
- Build artifacts uploaded for debugging
- Updated to actions@v4/v5

### 3.4 `docs.yml` — Redesigned

```yaml
name: Documentation

on:
  push:
    branches: [master]  # Only deploy docs on master merge
  pull_request:
    branches: [master, dev]  # Build (but don't deploy) on PRs

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: |
          pip install sphinx
          pip install -r docs/requirements.txt
      - name: Build documentation
        run: sphinx-build docs docs/_build/html -W  # -W makes warnings into errors
      - uses: actions/upload-artifact@v4
        with:
          name: documentation
          path: docs/_build/html/

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/master'
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: |
          pip install sphinx
          pip install -r docs/requirements.txt
      - run: sphinx-build docs docs/_build/html
      - uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_build/html
```

**Key changes:**
- Docs deploy ONLY on push to master (not on every PR)
- PRs still build docs to catch errors
- Replaced deprecated `ad-m/github-push-action@v0.6.0` with `peaceiris/actions-gh-pages@v4`
- Removed redundant `git clone` hack
- Added `-W` flag to make Sphinx warnings into errors

### 3.5 `codeql-analysis.yml` — Redesigned

```yaml
name: CodeQL

on:
  push:
    branches: [dev, master]
  pull_request:
    branches: [dev, master]
  schedule:
    - cron: '0 3 * * 1'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
```

**Key changes:**
- Updated from `@v1` (removed/broken) to `@v3`
- Updated `actions/checkout` from v2 to v4
- Removed manual `git checkout HEAD^2` hack
- Added explicit `languages: python`
- Added `master` branch to triggers
- Removed Autobuild step (not needed for Python)
- Added proper `security-events: write` permission

---

## 4. Version Management Fix

### 4.1 Single-Source Version

Currently version is in 4 places. Fix with single source in `pyproject.toml`:

```toml
[project]
name = "lazypredict"
version = "0.2.16"  # Single source of truth
```

Then in `__init__.py`:
```python
from importlib.metadata import version
__version__ = version("lazypredict")
```

And remove version from `setup.py` entirely (read from `pyproject.toml`).

### 4.2 Release Please Configuration

Add `.release-please-manifest.json`:
```json
{
  ".": "0.2.16"
}
```

Add `release-please-config.json`:
```json
{
  "packages": {
    ".": {
      "release-type": "python",
      "package-name": "lazypredict",
      "bump-minor-pre-major": true,
      "bump-patch-for-minor-pre-major": true,
      "extra-files": ["setup.py", "lazypredict/__init__.py"]
    }
  }
}
```

---

## 5. Required One-Time Setup

### 5.1 PyPI Trusted Publishers (replaces API tokens)

1. Go to https://pypi.org/manage/project/lazypredict/settings/publishing/
2. Add a new "pending publisher":
   - **Owner:** `shankarpandala`
   - **Repository:** `lazypredict`
   - **Workflow:** `publish.yml`
   - **Environment:** `pypi`
3. Repeat for TestPyPI at https://test.pypi.org

### 5.2 GitHub Environments

1. Go to repo Settings → Environments
2. Create `pypi` environment with:
   - Required reviewers (optional, for approval gate)
   - Deployment branch: `master` only
3. Create `testpypi` environment (no restrictions needed)

### 5.3 Branch Protection Rules

1. Go to repo Settings → Branches → Add rule for `master`:
   - Require pull request reviews (1 reviewer)
   - Require status checks to pass: `lint`, `test-fast`, `test-full`
   - Require branches to be up to date
   - Do not allow bypassing
2. Add rule for `dev`:
   - Require status checks to pass: `lint`, `test-fast`
   - Allow bypassing for maintainers

---

## 6. End-to-End Release Flow (After Implementation)

```
Developer                    GitHub Actions                PyPI
─────────                    ──────────────                ────
1. Write code on feature branch
2. Use conventional commits:
   "feat: add XGBoost timeout"
   "fix: handle empty DataFrame"
   "feat!: new fit() API"

3. Open PR to dev        →   CI runs (lint + 2 test jobs)
                              ✅ or ❌ status on PR

4. Merge to dev          →   CI runs again on dev

5. Open PR dev→master    →   Full CI matrix (9 jobs)
                              ✅ or ❌ status on PR

6. Merge to master       →   release-please scans commits
                              Creates "Release PR" with:
                              • Bumped version (0.2.17)
                              • Updated CHANGELOG.md

7. Review & merge         →   release-please creates:
   Release PR                 • GitHub Release "v0.2.17"
                              • Git tag "v0.2.17"

                         →   publish.yml triggers:
                              • Runs tests (gate)
                              • Builds wheel + sdist
                              • twine check
                              • Publishes to TestPyPI    →  test.pypi.org
                              • Publishes to PyPI        →  pypi.org ✅

                         →   docs.yml triggers:
                              • Builds Sphinx docs
                              • Deploys to gh-pages
```

**You never touch the GitHub Releases page again.** Just merge PRs with good commit messages.

---

## 7. Conventional Commit Cheatsheet

For release-please to work, use these commit prefixes:

| Prefix | Version Bump | Example |
|--------|-------------|---------|
| `fix:` | Patch (0.0.x) | `fix: handle empty DataFrame in fit()` |
| `feat:` | Minor (0.x.0) | `feat: add LightGBM timeout support` |
| `feat!:` or `BREAKING CHANGE:` | Major (x.0.0) | `feat!: change fit() return type` |
| `docs:` | No release | `docs: update README examples` |
| `chore:` | No release | `chore: update CI workflow` |
| `test:` | No release | `test: add negative test cases` |
| `refactor:` | No release | `refactor: extract base class` |

---

## 8. Migration Checklist

- [ ] Update `ci.yml` (remove `|| true`, add caching, split fast/full)
- [ ] Update `publish.yml` (add test gate, twine check, TestPyPI, OIDC)
- [ ] Update `docs.yml` (deploy only on master, replace deprecated action)
- [ ] Update `codeql-analysis.yml` (v1→v3, add master branch)
- [ ] Add `release-please.yml` workflow
- [ ] Add `.release-please-manifest.json`
- [ ] Add `release-please-config.json`
- [ ] Configure PyPI Trusted Publishers (manual, pypi.org UI)
- [ ] Create GitHub Environments `pypi` and `testpypi` (manual, repo settings)
- [ ] Set branch protection rules (manual, repo settings)
- [ ] Adopt conventional commits going forward
- [ ] Fix `meta.yaml` version (0.2.15 → 0.2.16)
- [ ] Remove `pytest-runner` from `requirements.txt`

---

## 9. Issue Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| CI (ci.yml) | 1 | 1 | 3 | 0 | 5 |
| Publishing (publish.yml) | 0 | 3 | 3 | 2 | 8 |
| Docs (docs.yml) | 0 | 1 | 2 | 1 | 4 |
| CodeQL | 1 | 0 | 1 | 2 | 4 |
| Version Management | 0 | 2 | 1 | 1 | 4 |
| Branch Strategy | 0 | 3 | 0 | 1 | 4 |
| **Total** | **2** | **10** | **10** | **7** | **29** |
