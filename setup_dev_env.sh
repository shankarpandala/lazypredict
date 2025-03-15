#!/bin/bash
# Setup script for lazypredict development environment

echo "Setting up lazypredict development environment..."

# 1. Ensure required dependencies are installed
echo "Installing dependencies..."
pip install -r requirements.txt
pip install pytest mlflow pytest-cov

# 2. Install the package in development mode (will refresh on changes)
echo "Installing lazypredict in development mode..."
pip install -e .

echo "Setup complete! You can now run examples and tests."
echo ""
echo "To run all tests: python -m tests.run_tests"
echo "To run specific tests: python -m tests.run_tests test_models"
echo "To run examples: python examples/classification_example.py" 