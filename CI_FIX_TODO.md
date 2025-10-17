# CI Fix & Python 3.14 Support - TODO Checklist

**Status:** 0/18 completed | **Priority:** CRITICAL
**Estimated Time:** 4-6 hours total
**Last Updated:** 2025-10-17

---

## 🚨 CRITICAL FIXES (Priority 1) - Do These First

### ✅ TODO 1: Update CI Python Version Matrix
**File:** `.github/workflows/ci.yml:22`
**Status:** ⏳ Pending
**Time:** 2 minutes
**Complexity:** Easy

**Current:**
```yaml
python-version: ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13']
```

**New (with Python 3.14):**
```yaml
python-version: ['3.9', '3.10', '3.11', '3.12', '3.13', '3.14']
```

**Decision Required:**
- [ ] Option A: Drop Python 3.8 completely (Recommended)
- [ ] Option B: Keep Python 3.8 with legacy requirements (see TODO 2)

**Notes:** Python 3.8 reached end-of-life in October 2024

---

### ✅ TODO 2: Create requirements-legacy.txt (Optional - Only if keeping Python 3.8-3.9)
**File:** `requirements-legacy.txt` (new file)
**Status:** ⏳ Pending
**Time:** 10 minutes
**Complexity:** Medium

**Create new file:**
```txt
# LazyPredict Dependencies - Python 3.8-3.9 Legacy Support
# These versions are the last to support Python 3.8

click>=8.1.0,<9.0.0
scikit-learn>=1.3.0,<1.4.0   # Last version supporting Python 3.8
pandas>=2.0.0,<2.1.0          # Last version supporting Python 3.8
numpy>=1.24.0,<1.25.0         # Last version supporting Python 3.8
tqdm>=4.65.0,<5.0.0
joblib>=1.3.0,<2.0.0
lightgbm>=4.0.0,<5.0.0
xgboost>=2.0.0,<3.0.0
mlflow>=2.10.0,<3.0.0
shap>=0.44.0,<1.0.0
pyarrow>=13.0.0,<14.0.0
```

**Testing:**
```bash
# Create Python 3.8 virtual environment
python3.8 -m venv venv38
source venv38/bin/activate  # Windows: venv38\Scripts\activate
pip install -r requirements-legacy.txt
pytest tests/ -v
```

---

### ✅ TODO 3: Update requirements.txt for Python 3.10-3.14
**File:** `requirements.txt`
**Status:** ⏳ Pending
**Time:** 15 minutes
**Complexity:** Medium

**Current Issue:** Versions are TOO restrictive (e.g., `>=1.7.2` which may not exist for Python 3.14 yet)

**Updated Version (Recommended):**
```txt
# LazyPredict Dependencies - Python 3.10-3.14 Support
# Using wider version ranges for better compatibility

# CLI Framework
click>=8.1.0,<9.0.0

# Core ML Libraries (wider ranges for Python 3.14)
scikit-learn>=1.3.0,<2.0.0    # Changed from >=1.7.2
pandas>=2.0.0,<3.0.0           # Changed from >=2.3.3
numpy>=1.24.0,<3.0.0           # Changed from >=2.3.3

# Progress Bars & Visualization
tqdm>=4.65.0,<5.0.0            # Changed from >=4.67.0
rich>=13.0.0,<15.0.0           # Changed from >=14.1.0

# Parallel Computing
joblib>=1.3.0,<2.0.0           # Changed from >=1.5.2

# Gradient Boosting Libraries
lightgbm>=4.0.0,<5.0.0         # Changed from >=4.6.0
xgboost>=2.0.0,<4.0.0          # Changed from >=3.0.4

# Experiment Tracking
mlflow>=2.10.0,<4.0.0          # Changed from >=2.22.2

# Model Explainability
shap>=0.44.0,<1.0.0            # Changed from >=0.48.0

# Performance Enhancements
pyarrow>=13.0.0,<20.0.0        # Changed from >=19.0.0
```

**Rationale:** Wider version ranges allow pip to find compatible versions for Python 3.14

**Testing:**
```bash
# Test with Python 3.14
python3.14 -m venv venv314
source venv314/bin/activate  # Windows: venv314\Scripts\activate
pip install -r requirements.txt
python -c "import sklearn; import pandas; import numpy; print('All imports successful')"
pytest tests/ -v
```

---

### ✅ TODO 4: Add Python 3.14 to CI Matrix
**File:** `.github/workflows/ci.yml:22`
**Status:** ⏳ Pending
**Time:** 1 minute
**Complexity:** Easy

**Update matrix to include 3.14:**
```yaml
python-version: ['3.9', '3.10', '3.11', '3.12', '3.13', '3.14']
```

**Also check:** Python 3.14 may be in beta/rc, so you might need:
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    allow-prereleases: true  # Add this for Python 3.14 if needed
```

---

### ✅ TODO 5: Update CI Workflow with Conditional Requirements
**File:** `.github/workflows/ci.yml:46-50`
**Status:** ⏳ Pending
**Time:** 10 minutes
**Complexity:** Medium

**Replace entire "Test with pytest" step:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip setuptools wheel

    # Install appropriate requirements based on Python version
    if [[ "${{ matrix.python-version }}" == "3.8" ]] || [[ "${{ matrix.python-version }}" == "3.9" ]]; then
      echo "Installing legacy requirements for Python ${{ matrix.python-version }}"
      python -m pip install -r requirements-legacy.txt
    else
      echo "Installing modern requirements for Python ${{ matrix.python-version }}"
      python -m pip install -r requirements.txt
    fi

    # Install test dependencies
    python -m pip install pytest pytest-cov pytest-xdist pytest-timeout
  shell: bash

- name: Run tests with pytest
  run: |
    pytest --cov=lazypredict \
           --cov-report=xml \
           --cov-report=html \
           --cov-report=term \
           -v \
           -n auto \
           --timeout=300
  shell: bash
```

**Note:** The `shell: bash` ensures this works on Windows runners

---

### ✅ TODO 6: Update setup.py for Python 3.14
**File:** `setup.py:25`
**Status:** ⏳ Pending
**Time:** 3 minutes
**Complexity:** Easy

**Update python_requires:**
```python
python_requires=">=3.9,<3.15",  # Changed from >=3.9,<3.15 (already correct!)
```

**Verify classifiers include 3.14:**
```python
classifiers=[
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",  # ✅ Already present!
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
],
```

**Good news:** Your setup.py already supports 3.14! ✅

---

### ✅ TODO 7: Update setup.py python_requires to Match CI
**File:** `setup.py:25`
**Status:** ⏳ Pending
**Time:** 1 minute
**Complexity:** Easy

**Decision:**
- If dropping Python 3.8: Keep `python_requires=">=3.9,<3.15"` ✅ (Already correct!)
- If keeping Python 3.8: Change to `python_requires=">=3.8,<3.15"`

**Action:** Verify this matches your CI matrix decision from TODO 1

---

## 🔧 IMPORTANT FIXES (Priority 2) - Modernization

### ✅ TODO 8: Update CodeQL Workflow Actions
**File:** `.github/workflows/codeql-analysis.yml`
**Status:** ⏳ Pending
**Time:** 5 minutes
**Complexity:** Easy

**Changes needed:**

**Line 19 - Update checkout:**
```yaml
- uses: actions/checkout@v2
+ uses: actions/checkout@v4
```

**Lines 31, 40, 54 - Update CodeQL actions:**
```yaml
- uses: github/codeql-action/init@v1
+ uses: github/codeql-action/init@v3

- uses: github/codeql-action/autobuild@v1
+ uses: github/codeql-action/autobuild@v3

- uses: github/codeql-action/analyze@v1
+ uses: github/codeql-action/analyze@v3
```

**Line 27 - Remove deprecated checkout (no longer needed):**
```yaml
# DELETE THIS BLOCK:
- run: git checkout HEAD^2
  if: ${{ github.event_name == 'pull_request' }}
```

---

### ✅ TODO 9: Update Publish Workflow Actions
**File:** `.github/workflows/publish.yml`
**Status:** ⏳ Pending
**Time:** 5 minutes
**Complexity:** Easy

**Changes:**
```yaml
# Line 17: Update checkout
- uses: actions/checkout@v2
+ uses: actions/checkout@v4

# Lines 20-22: Update Python setup
- uses: actions/setup-python@v2
+ uses: actions/setup-python@v5
  with:
-   python-version: '3.x'
+   python-version: '3.11'  # Use specific stable version
```

**Also verify:** Line 42 uses `pypa/gh-action-pypi-publish@release/v1` which is current ✅

---

### ✅ TODO 10: Fix Docs Workflow Git Push
**File:** `.github/workflows/docs.yml:43-48`
**Status:** ⏳ Pending
**Time:** 5 minutes
**Complexity:** Easy

**Replace:**
```yaml
- name: Push changes
  uses: ad-m/github-push-action@v0.6.0
  with:
    branch: gh-pages
    directory: gh-pages
    github_token: ${{ secrets.GITHUB_TOKEN }}
```

**With modern approach:**
```yaml
- name: Push changes
  run: |
    cd gh-pages
    git push https://x-access-token:${{ secrets.GITHUB_TOKEN }}@github.com/${{ github.repository }}.git gh-pages
  continue-on-error: true  # Don't fail if no changes
```

**Or use modern action:**
```yaml
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./docs/_build/html
    publish_branch: gh-pages
```

---

## ⚡ OPTIMIZATION (Priority 3) - Performance

### ✅ TODO 11: Add Pip Dependency Caching
**File:** `.github/workflows/ci.yml` (after line 32)
**Status:** ⏳ Pending
**Time:** 5 minutes
**Complexity:** Easy
**Benefit:** Saves ~30-60 seconds per CI job

**Add after "Set up Python" step:**
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}

- name: Cache pip dependencies  # ← ADD THIS
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/Library/Caches/pip
      ~\AppData\Local\pip\Cache
    key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-${{ matrix.python-version }}-
      ${{ runner.os }}-pip-

- name: Install dependencies
  run: |
    # ... rest of install
```

---

### ✅ TODO 12: Create Separate Linting Job
**File:** `.github/workflows/ci.yml:16`
**Status:** ⏳ Pending
**Time:** 10 minutes
**Complexity:** Medium
**Benefit:** Faster feedback, runs linting once instead of 18 times

**Add BEFORE the `build` job:**
```yaml
jobs:
  lint:
    name: Lint and Format Check
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Cache pip dependencies
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: lint-pip-${{ hashFiles('requirements*.txt') }}

    - name: Install linting tools
      run: |
        python -m pip install --upgrade pip
        pip install flake8 black isort ruff

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127

    - name: Check code formatting with black
      run: |
        black --check lazypredict tests
      continue-on-error: true

    - name: Check import sorting with isort
      run: |
        isort --check-only lazypredict tests
      continue-on-error: true

  build:
    name: Test Python ${{ matrix.python-version }} on ${{ matrix.os }}
    needs: lint  # ← ADD THIS - build only runs if lint passes
    runs-on: ${{ matrix.os }}
    # ... rest of build job
```

**Then REMOVE flake8 steps from build job (lines 39-44)**

---

### ✅ TODO 13: Add Matrix Strategy with Requirements Mapping
**File:** `.github/workflows/ci.yml:19-23`
**Status:** ⏳ Pending
**Time:** 10 minutes
**Complexity:** Medium

**Replace strategy section:**
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.9', '3.10', '3.11', '3.12', '3.13', '3.14']
    include:
      # Python 3.9 uses legacy requirements
      - python-version: '3.9'
        requirements-file: 'requirements-legacy.txt'
      # Python 3.10+ uses modern requirements
      - python-version: '3.10'
        requirements-file: 'requirements.txt'
      - python-version: '3.11'
        requirements-file: 'requirements.txt'
      - python-version: '3.12'
        requirements-file: 'requirements.txt'
      - python-version: '3.13'
        requirements-file: 'requirements.txt'
      - python-version: '3.14'
        requirements-file: 'requirements.txt'
    exclude:
      # Optional: Reduce matrix size for faster CI
      # Only test Python 3.9 and 3.14 on Ubuntu
      - os: macos-latest
        python-version: '3.9'
      - os: macos-latest
        python-version: '3.14'
      - os: windows-latest
        python-version: '3.9'
      - os: windows-latest
        python-version: '3.14'
  fail-fast: false
```

**Then update install step:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r ${{ matrix.requirements-file || 'requirements.txt' }}
    python -m pip install pytest pytest-cov pytest-xdist pytest-timeout
  shell: bash
```

---

## 🧪 TESTING (Priority 4) - Validation

### ✅ TODO 14: Test Python 3.14 Compatibility Locally
**Status:** ⏳ Pending
**Time:** 20 minutes
**Complexity:** Medium

**Steps:**

1. **Install Python 3.14:**
   ```bash
   # Check if available
   python3.14 --version

   # If not, install from python.org or pyenv
   pyenv install 3.14.0
   ```

2. **Create virtual environment:**
   ```bash
   python3.14 -m venv venv314
   source venv314/bin/activate  # Windows: venv314\Scripts\activate
   ```

3. **Test dependency installation:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run imports test:**
   ```bash
   python -c "
   import sklearn
   import pandas
   import numpy
   import lightgbm
   import xgboost
   import mlflow
   import shap
   print('✅ All imports successful!')
   print(f'scikit-learn: {sklearn.__version__}')
   print(f'pandas: {pandas.__version__}')
   print(f'numpy: {numpy.__version__}')
   "
   ```

5. **Run test suite:**
   ```bash
   pip install pytest pytest-cov pytest-xdist pytest-timeout
   pytest tests/ -v --tb=short
   ```

**Document results:**
- [ ] All dependencies install successfully
- [ ] All imports work
- [ ] All tests pass
- [ ] No deprecation warnings

---

### ✅ TODO 15: Verify All Tests Pass with Python 3.14
**Status:** ⏳ Pending
**Time:** 15 minutes
**Complexity:** Easy

**Full test suite:**
```bash
# Activate Python 3.14 environment
source venv314/bin/activate

# Run comprehensive tests
pytest tests/ -v \
       --cov=lazypredict \
       --cov-report=term-missing \
       --cov-report=html \
       -n auto \
       --timeout=300

# Check coverage report
open htmlcov/index.html  # Or start htmlcov/index.html on Windows
```

**Expected results:**
- [ ] All tests pass
- [ ] Coverage ≥ 75%
- [ ] No failures in test_supervised.py
- [ ] No failures in test_supervised_comprehensive.py
- [ ] CLI tests pass (test_cli.py)
- [ ] Explainer tests pass (test_explainer.py)

**If tests fail:**
- Document which tests fail
- Check if it's a Python 3.14 specific issue
- Create issues for Python 3.14 incompatibilities

---

### ✅ TODO 16: Update requirements_dev.txt for Python 3.14
**File:** `requirements_dev.txt`
**Status:** ⏳ Pending
**Time:** 10 minutes
**Complexity:** Easy

**Review and widen version ranges:**
```txt
# LazyPredict Development Dependencies - Python 3.9-3.14 Support

# Inherit production dependencies
-r requirements.txt

# Testing Framework (widen ranges)
pytest>=7.4.0,<9.0.0          # Changed from >=7.4.4
pytest-cov>=4.1.0,<6.0.0      # Already OK
pytest-xdist>=3.3.0,<4.0.0    # Changed from >=3.6.0
pytest-benchmark>=4.0.0,<6.0.0 # Changed from >=5.0.0
pytest-timeout>=2.1.0,<3.0.0  # Changed from >=2.3.0

# Code Formatting (widen ranges)
black>=23.0.0,<25.0.0         # Changed from >=24.0.0
isort>=5.12.0,<6.0.0          # Changed from >=5.13.0

# Linting & Quality (widen ranges)
flake8>=6.0.0,<8.0.0          # Changed from >=7.0.0
pylint>=3.0.0,<4.0.0          # Changed from >=3.3.0
bandit>=1.7.0,<2.0.0          # Changed from >=1.8.0
ruff>=0.1.0,<1.0.0            # Changed from >=0.8.0

# Type Checking
mypy>=1.5.0,<2.0.0            # Changed from >=1.18.0

# Documentation
myst-parser>=2.0.0,<4.0.0     # Changed from >=3.0.0
sphinx>=7.0.0,<8.0.0          # Changed from >=7.4.0
sphinx-rtd-theme>=2.0.0,<4.0.0 # Changed from >=3.0.0

# Performance Profiling
memory-profiler>=0.61.0,<1.0.0
line-profiler>=4.0.0,<5.0.0   # Changed from >=4.1.0
py-spy>=0.3.0,<1.0.0          # Changed from >=0.4.0
scalene>=1.5.0,<2.0.0

# Development Utilities
ipython>=8.12.0,<9.0.0        # Changed from >=8.30.0
jupyter>=1.0.0,<2.0.0         # Already OK
```

**Test:**
```bash
source venv314/bin/activate
pip install -r requirements_dev.txt
```

---

## 📚 DOCUMENTATION (Priority 5) - Polish

### ✅ TODO 17: Update CLAUDE.md with Python Version Notes
**File:** `CLAUDE.md:1-50`
**Status:** ⏳ Pending
**Time:** 10 minutes
**Complexity:** Easy

**Add section after line 9 (after Key Features):**

```markdown
**Python Version Support:**
- Python 3.9-3.14: Fully supported with modern dependencies
- Python 3.8: End-of-life (October 2024) - Use `requirements-legacy.txt` if needed
- Recommended: Python 3.11 or 3.12 for best stability

**Dependency Strategy:**
- `requirements.txt`: Python 3.10-3.14 (modern stack)
- `requirements-legacy.txt`: Python 3.8-3.9 (frozen older versions)
- `requirements_dev.txt`: Development tools (all Python versions)
```

**Update Testing section (around line 40):**
```markdown
### Testing

**Python Version Compatibility:**
```bash
# Python 3.10-3.14 (Modern)
pip install -r requirements.txt
pytest -v

# Python 3.8-3.9 (Legacy - if needed)
pip install -r requirements-legacy.txt
pytest -v
```

**Run all tests:**
```bash
# Basic test run
pytest -v

# With coverage
pytest --cov=lazypredict --cov-report=html --cov-report=term -v

# Parallel execution (faster)
pytest -n auto -v

# Specific Python version (example: 3.14)
python3.14 -m pytest -v
```
```

---

### ✅ TODO 18: Add CI Status Badges to README
**File:** `README.md:1`
**Status:** ⏳ Pending
**Time:** 5 minutes
**Complexity:** Easy

**Add at the top of README.md (after title):**
```markdown
# LazyPredict

[![CI](https://github.com/shankarpandala/lazypredict/workflows/CI/badge.svg?branch=dev)](https://github.com/shankarpandala/lazypredict/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shankarpandala/lazypredict/branch/dev/graph/badge.svg)](https://codecov.io/gh/shankarpandala/lazypredict)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/lazypredict.svg)](https://badge.fury.io/py/lazypredict)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/lazypredict)](https://pepy.tech/project/lazypredict)

Lazy Predict helps you build lots of basic models without much code...
```

---

## 📊 PROGRESS TRACKER

### Summary
- **Total Tasks:** 18
- **Completed:** 0
- **In Progress:** 0
- **Pending:** 18

### By Priority
- **🚨 Critical (P1):** 7 tasks - Fixes CI immediately
- **🔧 Important (P2):** 3 tasks - Modernizes workflows
- **⚡ Optimization (P3):** 3 tasks - Improves performance
- **🧪 Testing (P4):** 3 tasks - Validates changes
- **📚 Documentation (P5):** 2 tasks - Polish

### Estimated Time
- **Quick Fix (P1 only):** 45 minutes
- **Full Fix (P1 + P2):** 1.5 hours
- **Complete (All):** 4-6 hours

---

## 🎯 RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Emergency Fix (45 min) - Get CI Green
```bash
✓ TODO 1: Update CI matrix (drop 3.8, add 3.14)
✓ TODO 3: Widen requirements.txt version ranges
✓ TODO 6: Verify setup.py (already correct)
✓ TODO 5: Add conditional requirements (simplified)
```

### Phase 2: Stabilization (1 hour) - Ensure Quality
```bash
✓ TODO 14: Test Python 3.14 locally
✓ TODO 15: Verify all tests pass
✓ TODO 16: Update requirements_dev.txt
✓ TODO 8: Update CodeQL workflow
```

### Phase 3: Optimization (1 hour) - Improve Performance
```bash
✓ TODO 11: Add pip caching
✓ TODO 12: Separate linting job
✓ TODO 9: Update Publish workflow
✓ TODO 10: Fix Docs workflow
```

### Phase 4: Polish (1 hour) - Documentation & Enhancement
```bash
✓ TODO 17: Update CLAUDE.md
✓ TODO 18: Add CI badges
✓ TODO 13: Add matrix strategy mapping
✓ TODO 2: Create legacy requirements (if needed)
```

---

## 🚀 QUICK START (Fastest Path to Green CI)

If you want CI to pass **immediately**, run these commands:

```bash
# 1. Update CI to remove Python 3.8 and add 3.14
# Edit .github/workflows/ci.yml line 22:
# python-version: ['3.9', '3.10', '3.11', '3.12', '3.13', '3.14']

# 2. Widen requirements.txt ranges
sed -i 's/scikit-learn>=1.7.2/scikit-learn>=1.3.0/' requirements.txt
sed -i 's/pandas>=2.3.3/pandas>=2.0.0/' requirements.txt
sed -i 's/numpy>=2.3.3/numpy>=1.24.0/' requirements.txt
sed -i 's/tqdm>=4.67.0/tqdm>=4.65.0/' requirements.txt
sed -i 's/rich>=14.1.0/rich>=13.0.0/' requirements.txt
sed -i 's/joblib>=1.5.2/joblib>=1.3.0/' requirements.txt
sed -i 's/lightgbm>=4.6.0/lightgbm>=4.0.0/' requirements.txt
sed -i 's/xgboost>=3.0.4/xgboost>=2.0.0/' requirements.txt
sed -i 's/mlflow>=2.22.2/mlflow>=2.10.0/' requirements.txt
sed -i 's/shap>=0.48.0/shap>=0.44.0/' requirements.txt
sed -i 's/pyarrow>=19.0.0/pyarrow>=13.0.0/' requirements.txt

# 3. Commit and push
git add .github/workflows/ci.yml requirements.txt
git commit -m "fix(ci): Update Python versions (3.9-3.14) and widen dependency ranges"
git push origin dev

# 4. Watch CI at: https://github.com/shankarpandala/lazypredict/actions
```

---

## ❓ DECISION POINTS

You need to decide on these before implementing:

### Decision 1: Python 3.8 Support
- [ ] **Option A (Recommended):** Drop Python 3.8 entirely
  - Pros: Simpler, cleaner, Python 3.8 is EOL
  - Cons: May break existing users on 3.8
  - Action: Skip TODO 2 (no legacy file needed)

- [ ] **Option B:** Keep Python 3.8 with legacy requirements
  - Pros: Backward compatible
  - Cons: More maintenance, two requirements files
  - Action: Complete TODO 2

### Decision 2: CI Matrix Size
Current: 18 jobs (3 OS × 6 Python versions)

- [ ] **Option A (Recommended):** Reduced matrix with strategic coverage
  - Test all Python versions on Ubuntu only
  - Test Python 3.11 on all OS (most stable)
  - Result: ~8 jobs instead of 18 (saves $$)

- [ ] **Option B:** Full matrix
  - Test everything everywhere
  - Result: 18 jobs (current)
  - Cost: ~2x longer CI time, more GitHub Actions minutes

### Decision 3: Version Range Strategy
- [ ] **Option A (Recommended):** Wide ranges (>=1.3.0,<2.0.0)
  - Pros: Better compatibility, easier upgrades
  - Cons: Less predictable, potential breaking changes

- [ ] **Option B:** Pinned versions (==1.7.2)
  - Pros: Fully reproducible
  - Cons: Breaks on new Python versions, constant updates needed

---

## 🐛 KNOWN ISSUES TO WATCH FOR

1. **Python 3.14 Beta Status**
   - Python 3.14 is still in development (final release October 2025)
   - Some packages may not have wheels yet
   - CI might need `allow-prereleases: true`

2. **NumPy 2.0 Breaking Changes**
   - NumPy 2.0+ has API changes
   - Some older code might break
   - Test thoroughly with NumPy 2.x

3. **scikit-learn API Deprecations**
   - scikit-learn 1.5+ deprecated some parameters
   - Check for FutureWarnings in tests
   - Update code if needed

4. **LightGBM/XGBoost Compilation**
   - May not have Python 3.14 wheels yet
   - Might need to compile from source (slow)
   - Consider falling back to older versions temporarily

---

## 📞 SUPPORT & NEXT STEPS

**After completing each TODO:**
- [ ] Run tests locally
- [ ] Commit with descriptive message
- [ ] Push to dev branch
- [ ] Monitor CI results
- [ ] Update this document with ✅ or notes

**If you get stuck:**
1. Check the specific error message in CI logs
2. Search for the error on GitHub/StackOverflow
3. Try with a single Python version first
4. Reach out with specific error details

**Success Criteria:**
- ✅ All CI jobs pass
- ✅ Python 3.9-3.14 supported
- ✅ Tests pass on Ubuntu, macOS, Windows
- ✅ Code coverage ≥ 75%
- ✅ No critical flake8 errors

---

**Last Updated:** 2025-10-17
**Document Version:** 1.0
**Status:** Ready for implementation
