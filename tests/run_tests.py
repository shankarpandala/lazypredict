#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test runner for lazypredict package.
This script runs all the tests for the package with appropriate settings.
"""

import os
import sys
import unittest
import warnings

# Add the parent directory to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# Ignore warnings during tests
warnings.filterwarnings("ignore")


def run_all_tests():
    """Run all tests in the package."""
    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover("tests")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result


def run_specific_test(test_name):
    """Run a specific test module."""
    if not test_name.startswith("test_"):
        test_name = f"test_{test_name}"
    if not test_name.endswith(".py"):
        test_name = f"{test_name}.py"

    # Get the module name without extension
    module_name = test_name[:-3]

    # Import the module
    try:
        module = __import__(f"tests.{module_name}", fromlist=["tests"])
        test_suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)
        return result
    except ImportError:
        print(f"Could not import test module {module_name}")
        return None


if __name__ == "__main__":
    # If a test name is provided, run that specific test
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        print(f"Running test: {test_name}")
        result = run_specific_test(test_name)
    else:
        # Otherwise run all tests
        print("Running all tests")
        result = run_all_tests()

    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())
