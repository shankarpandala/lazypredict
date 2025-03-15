#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

# Core requirements
core_requirements = [
    "numpy>=1.17.0",
    "pandas>=1.0.0",
    "tqdm>=4.45.0",
    "scikit-learn>=1.0.0",
    "xgboost>=1.0.0",
    "lightgbm>=3.0.0",
]

# Optional dependencies
mlflow_requirements = ["mlflow>=1.0.0"]
optuna_requirements = ["optuna>=2.0.0"]
gpu_requirements = ["torch>=1.0.0", "cuml>=21.0.0"]
survival_requirements = ["scikit-survival>=0.15.0"]

# Development requirements
dev_requirements = [
    "pytest>=5.0.0",
    "pytest-runner>=5.0.0",
    "black>=20.8b1",
    "isort>=5.0.0",
    "mypy>=0.800",
    "flake8>=3.8.0",
    "sphinx>=3.0.0",
    "twine>=3.0.0",
]

setup(
    author="Shankar Rao Pandala",
    author_email="shankar.pandala@live.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Lazy Predict helps build a lot of basic models without much code and helps understand which models work better without any parameter tuning",
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["lazypredict=lazypredict.cli:main",],},
    install_requires=core_requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="lazypredict, machine learning, data science, automated machine learning",
    name="lazypredict",
    packages=find_packages(include=["lazypredict", "lazypredict.*"]),
    setup_requires=["pytest-runner"],
    test_suite="tests",
    tests_require=["pytest>=5.0.0"],
    url="https://github.com/shankarpandala/lazypredict",
    version='0.3.0',
    zip_safe=False,
    extras_require={
        "dev": dev_requirements,
        "mlflow": mlflow_requirements,
        "optuna": optuna_requirements,
        "gpu": gpu_requirements,
        "survival": survival_requirements,
        "all": (
            mlflow_requirements 
            + optuna_requirements 
            + gpu_requirements 
            + survival_requirements
        ),
    },
)
