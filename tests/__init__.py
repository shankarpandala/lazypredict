# -*- coding: utf-8 -*-

"""
Test suite for lazypredict package.
This package contains various test modules for different components:

- test_models.py: Tests for specific model modules (classification, regression, etc.)
- test_base.py: Tests for base functionality, utilities, and core classes
- test_integration.py: Tests for compatibility between components and end-to-end functionality
- test_model_registry.py: Tests to verify all models are properly registered
"""

# Make imports easier for discovery
from .test_models import *
from .test_base import *
from .test_integration import *
from .test_model_registry import *
