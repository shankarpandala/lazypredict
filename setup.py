#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

requirements = [requirement for requirement in open('requirements.txt')]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Kumar Nityan Suman",
    author_email="nityan.suman@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    description="[Updated] Lazy Predict help build a lot of basic models without much code and helps understand which models works better without any parameter tuning",
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["lazypredict=lazypredict.cli:main",],},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords=["lazypredict", "lazypredict-nightly"],
    name="lazypredict-nightly",
    setup_requires=setup_requirements,
    test_suite="tests",
    url="https://github.com/nityansuman/lazypredict-nightly",
    version='0.3.0',
    zip_safe=False
)
