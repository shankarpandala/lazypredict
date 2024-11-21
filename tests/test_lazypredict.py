#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lazypredict` package."""

import pytest

from click.testing import CliRunner

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lazypredict import cli
from lazypredict.Supervised import LazyClassifier

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "lazypredict.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output

def test_hyperparameter_set():

    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=10, n_redundant=0, random_state=42
    )

    train_samples = 100  # Samples used for training the models
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=500 - train_samples,
    )

    clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            hyperparameters_dict={"DecisionTreeClassifier": {"max_depth": 4}})
    _ , _ = clf.fit(X_train, X_test, y_train, y_test)