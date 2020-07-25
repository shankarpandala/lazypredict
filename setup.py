#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "click>=7.1.2",
    "joblib>=0.16.0",
    "lightgbm>=2.3.1",
    "numpy>=1.19.1",
    "optuna>=1.5.0",
    "pandas>=1.0.5",
    "pytest>=5.4.3",
    "PyYAML>=5.3.1",
    "scikit-learn>=0.23.1",
    "scipy>=1.5.2",
    "six>=1.15.0",
    "tqdm>=4.48.0",
    "xgboost>=1.1.1",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Shankar Rao Pandala",
    author_email="shankar.pandala@live.com",
    python_requires=">=3.8, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="Lazy Predict help build a lot of basic models without much code and helps understand which models works better without any parameter tuning",
    entry_points={"console_scripts": ["lazypredict=lazypredict.cli:main",],},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="lazypredict",
    name="lazypredict",
    packages=find_packages(include=["lazypredict", "lazypredict.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/shankarpandala/lazypredict",
    version="0.2.7",
    zip_safe=False,
)
