#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run a full test cycle after code changes.
This reinstalls the package and runs tests.
"""

import os
import sys
import subprocess
import argparse

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=False)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    return True

def run_test_cycle(test_module=None, run_example=None):
    """Run a full test cycle"""
    print("Running test cycle for lazypredict...")
    
    # Reinstall the package
    print("\nReinstalling package...")
    if not run_command("pip install -e ."):
        print("Failed to reinstall package")
        return False
    
    # Run specified tests
    if test_module:
        print(f"\nRunning specified test module: {test_module}")
        if not run_command(f"python -m tests.run_tests {test_module}"):
            print("Tests failed")
            return False
    else:
        print("\nRunning all tests...")
        if not run_command("python -m tests.run_tests"):
            print("Tests failed")
            return False
    
    # Run example if specified
    if run_example:
        print(f"\nRunning example: {run_example}")
        if not run_command(f"python examples/{run_example}"):
            print("Example failed")
            return False
    
    print("\nTest cycle completed successfully!")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run a test cycle for lazypredict")
    parser.add_argument("--test", "-t", help="Specific test module to run")
    parser.add_argument("--example", "-e", help="Example to run (e.g., classification_example.py)")
    
    args = parser.parse_args()
    
    if run_test_cycle(args.test, args.example):
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main()) 