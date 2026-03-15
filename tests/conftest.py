"""Pytest configuration for lazypredict tests.

When torch is installed alongside scikit-learn or interpret (EBM), their
competing OpenMP runtimes can deadlock.  Setting thread-count env vars
before any compiled extension initialises OpenMP avoids the contention.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_START_METHOD", "fork")
