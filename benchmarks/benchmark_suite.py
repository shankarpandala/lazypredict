"""
Comprehensive benchmark suite for LazyPredict.

Tracks performance metrics over time to detect regressions.
"""

import time
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from lazypredict.Supervised import LazyClassifier, LazyRegressor


class BenchmarkRunner:
    """Run performance benchmarks and track results."""

    def __init__(self, output_dir="benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_classification_benchmark(self, n_samples=1000, n_features=20, n_jobs=1):
        """Benchmark classification performance."""
        # Generate data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Benchmark
        start = time.time()
        clf = LazyClassifier(verbose=0, ignore_warnings=True, n_jobs=n_jobs)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
        elapsed = time.time() - start

        result = {
            'benchmark': 'classification',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_jobs': n_jobs,
            'models_trained': len(models),
            'time_seconds': elapsed,
            'timestamp': datetime.datetime.now().isoformat()
        }

        self.results.append(result)
        return result

    def run_regression_benchmark(self, n_samples=1000, n_features=20, n_jobs=1):
        """Benchmark regression performance."""
        # Generate data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Benchmark
        start = time.time()
        reg = LazyRegressor(verbose=0, ignore_warnings=True, n_jobs=n_jobs)
        models, _ = reg.fit(X_train, X_test, y_train, y_test)
        elapsed = time.time() - start

        result = {
            'benchmark': 'regression',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_jobs': n_jobs,
            'models_trained': len(models),
            'time_seconds': elapsed,
            'timestamp': datetime.datetime.now().isoformat()
        }

        self.results.append(result)
        return result

    def run_scaling_benchmark(self):
        """Benchmark performance at different data scales."""
        sizes = [500, 1000, 2000, 5000]
        scaling_results = []

        for size in sizes:
            print(f"Benchmarking at {size} samples...")
            result = self.run_classification_benchmark(
                n_samples=size, n_features=20, n_jobs=-1
            )
            scaling_results.append(result)

        return scaling_results

    def run_parallel_benchmark(self):
        """Benchmark parallel vs sequential execution."""
        n_jobs_options = [1, 2, 4, -1]
        parallel_results = []

        for n_jobs in n_jobs_options:
            print(f"Benchmarking with n_jobs={n_jobs}...")
            result = self.run_classification_benchmark(
                n_samples=1000, n_features=20, n_jobs=n_jobs
            )
            parallel_results.append(result)

        return parallel_results

    def save_results(self, filename=None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {filepath}")
        return filepath

    def generate_report(self):
        """Generate a summary report."""
        if not self.results:
            print("No results to report")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "="*70)
        print("Benchmark Summary Report")
        print("="*70)

        # Overall statistics
        print(f"\nTotal benchmarks run: {len(df)}")
        print(f"Average time: {df['time_seconds'].mean():.2f}s")
        print(f"Median time: {df['time_seconds'].median():.2f}s")
        print(f"Min time: {df['time_seconds'].min():.2f}s")
        print(f"Max time: {df['time_seconds'].max():.2f}s")

        # By benchmark type
        print("\n" + "-"*70)
        print("By Benchmark Type:")
        print("-"*70)
        for benchmark_type in df['benchmark'].unique():
            subset = df[df['benchmark'] == benchmark_type]
            print(f"\n{benchmark_type.capitalize()}:")
            print(f"  Count: {len(subset)}")
            print(f"  Avg time: {subset['time_seconds'].mean():.2f}s")
            print(f"  Avg models: {subset['models_trained'].mean():.1f}")

        # Detailed results
        print("\n" + "-"*70)
        print("Detailed Results:")
        print("-"*70)
        print(df[['benchmark', 'n_samples', 'n_features', 'n_jobs', 'models_trained', 'time_seconds']].to_string(index=False))

        return df


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite."""
    runner = BenchmarkRunner()

    print("Starting comprehensive benchmark suite...")
    print("="*70)

    # Classification benchmarks
    print("\n1. Running classification benchmarks...")
    runner.run_classification_benchmark(n_samples=1000, n_features=20, n_jobs=1)
    runner.run_classification_benchmark(n_samples=1000, n_features=20, n_jobs=-1)

    # Regression benchmarks
    print("\n2. Running regression benchmarks...")
    runner.run_regression_benchmark(n_samples=1000, n_features=20, n_jobs=1)
    runner.run_regression_benchmark(n_samples=1000, n_features=20, n_jobs=-1)

    # Scaling benchmarks
    print("\n3. Running scaling benchmarks...")
    runner.run_scaling_benchmark()

    # Parallel benchmarks
    print("\n4. Running parallel execution benchmarks...")
    runner.run_parallel_benchmark()

    # Generate report
    print("\n" + "="*70)
    runner.generate_report()

    # Save results
    runner.save_results()

    print("\n" + "="*70)
    print("Benchmark suite complete!")
    print("="*70)


if __name__ == "__main__":
    run_comprehensive_benchmark()
