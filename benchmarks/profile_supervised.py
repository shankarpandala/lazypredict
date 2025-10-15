"""
Performance profiling script for LazyPredict Supervised Learning.

This script profiles LazyPredict to identify performance bottlenecks
and provides detailed timing information for optimization.
"""

import cProfile
import pstats
import io
from memory_profiler import profile as memory_profile
import time
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import LazyPredict
from lazypredict.Supervised import LazyClassifier, LazyRegressor


def generate_classification_data(n_samples=1000, n_features=20, n_informative=15):
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


def generate_regression_data(n_samples=1000, n_features=20, n_informative=15):
    """Generate synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


def profile_classification(n_jobs=1, n_samples=1000, n_features=20):
    """Profile LazyClassifier performance."""
    print(f"\n{'='*70}")
    print(f"Profiling LazyClassifier: n_jobs={n_jobs}, n_samples={n_samples}, n_features={n_features}")
    print(f"{'='*70}\n")

    # Generate data
    X_train, X_test, y_train, y_test = generate_classification_data(n_samples, n_features)

    # Create profiler
    profiler = cProfile.Profile()

    # Profile execution
    profiler.enable()
    start_time = time.time()

    clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=n_jobs)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    end_time = time.time()
    profiler.disable()

    # Print results
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Models trained: {len(models)}")
    print(f"\nTop 5 models by Balanced Accuracy:")
    print(models.head())

    # Print profiling statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print("\n" + "="*70)
    print("Top 20 functions by cumulative time:")
    print("="*70)
    print(s.getvalue())

    return models


def profile_regression(n_jobs=1, n_samples=1000, n_features=20):
    """Profile LazyRegressor performance."""
    print(f"\n{'='*70}")
    print(f"Profiling LazyRegressor: n_jobs={n_jobs}, n_samples={n_samples}, n_features={n_features}")
    print(f"{'='*70}\n")

    # Generate data
    X_train, X_test, y_train, y_test = generate_regression_data(n_samples, n_features)

    # Create profiler
    profiler = cProfile.Profile()

    # Profile execution
    profiler.enable()
    start_time = time.time()

    reg = LazyRegressor(verbose=0, ignore_warnings=True, n_jobs=n_jobs)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    end_time = time.time()
    profiler.disable()

    # Print results
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Models trained: {len(models)}")
    print(f"\nTop 5 models by Adjusted R-Squared:")
    print(models.head())

    # Print profiling statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print("\n" + "="*70)
    print("Top 20 functions by cumulative time:")
    print("="*70)
    print(s.getvalue())

    return models


def compare_parallel_vs_sequential():
    """Compare sequential vs parallel execution."""
    print("\n" + "="*70)
    print("Comparing Sequential vs Parallel Execution")
    print("="*70 + "\n")

    # Generate data
    X_train, X_test, y_train, y_test = generate_classification_data(n_samples=1000, n_features=20)

    # Sequential execution
    print("Testing Sequential Execution (n_jobs=1)...")
    start = time.time()
    clf_seq = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=1)
    models_seq, _ = clf_seq.fit(X_train, X_test, y_train, y_test)
    seq_time = time.time() - start
    print(f"Sequential time: {seq_time:.2f} seconds")

    # Parallel execution
    print("\nTesting Parallel Execution (n_jobs=-1)...")
    start = time.time()
    clf_par = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=-1)
    models_par, _ = clf_par.fit(X_train, X_test, y_train, y_test)
    par_time = time.time() - start
    print(f"Parallel time: {par_time:.2f} seconds")

    # Calculate speedup
    speedup = seq_time / par_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved: {seq_time - par_time:.2f} seconds")

    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': speedup,
        'models_count': len(models_seq)
    }


def profile_with_different_data_sizes():
    """Profile with different dataset sizes."""
    print("\n" + "="*70)
    print("Profiling with Different Data Sizes")
    print("="*70 + "\n")

    sizes = [500, 1000, 2000, 5000]
    results = []

    for size in sizes:
        print(f"\nTesting with {size} samples...")
        X_train, X_test, y_train, y_test = generate_classification_data(n_samples=size, n_features=20)

        start = time.time()
        clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=-1)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
        elapsed = time.time() - start

        results.append({
            'n_samples': size,
            'time': elapsed,
            'models_trained': len(models)
        })
        print(f"Time: {elapsed:.2f} seconds, Models: {len(models)}")

    return results


if __name__ == "__main__":
    import sys

    print("\n" + "="*70)
    print("LazyPredict Performance Profiling")
    print("="*70)

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ["all", "classification"]:
        # Profile classification
        profile_classification(n_jobs=1, n_samples=1000, n_features=20)

    if mode in ["all", "regression"]:
        # Profile regression
        profile_regression(n_jobs=1, n_samples=1000, n_features=20)

    if mode in ["all", "parallel"]:
        # Compare parallel vs sequential
        compare_parallel_vs_sequential()

    if mode in ["all", "scaling"]:
        # Profile with different data sizes
        profile_with_different_data_sizes()

    print("\n" + "="*70)
    print("Profiling Complete")
    print("="*70)
