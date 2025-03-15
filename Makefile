.PHONY: setup install test test-models test-base test-integration test-utils example-classification example-regression clean

# Setup development environment
setup:
	./setup_dev_env.sh

# Install the package in development mode
install:
	pip install -e .

# Run all tests
test:
	python -m tests.run_tests

# Run specific test modules
test-models:
	python -m tests.run_tests test_models

test-base:
	python -m tests.run_tests test_base

test-integration:
	python -m tests.run_tests test_integration

test-utils:
	python -m tests.run_tests test_utils

# Run examples
example-classification:
	python examples/classification_example.py

example-regression:
	python examples/regression_example.py

# Full test cycle - reinstall and run tests
test-cycle:
	python run_test_cycle.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
