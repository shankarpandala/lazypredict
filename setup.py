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
    author="Shankar Rao Pandala",
    author_email="shankar.pandala@live.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        'Programming Language :: Python :: 3.8',
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    description="Lazy Predict help build a lot of basic models without much code and helps understand which models works better without any parameter tuning",
    long_description_content_type="text/markdown",
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
    version='0.2.12',
    zip_safe=False,
)
