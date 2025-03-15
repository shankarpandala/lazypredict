#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for lazypredict development environment.
This script installs required dependencies and sets up the package for development.
"""

import os
import sys
import subprocess

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=False)
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    return True

def setup_environment():
    """Set up the development environment"""
    print("Setting up lazypredict development environment...")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install main dependencies")
        return False
    
    if not run_command("pip install pytest mlflow pytest-cov"):
        print("Failed to install test dependencies")
        return False
    
    # Install the package in development mode
    print("\nInstalling lazypredict in development mode...")
    if not run_command("pip install -e ."):
        print("Failed to install package in development mode")
        return False
    
    print("\nSetup complete! You can now run examples and tests.")
    print("\nTo run all tests: python -m tests.run_tests")
    print("To run specific tests: python -m tests.run_tests test_models")
    print("To run examples: python examples/classification_example.py")
    
    return True

def main():
    """Main function"""
    if setup_environment():
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main()) 