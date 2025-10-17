#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lazypredict` package."""

import pytest

from click.testing import CliRunner

from lazypredict import cli


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
    # Click returns 2 when no command is provided
    assert result.exit_code in [0, 2]
    assert "LazyPredict" in result.output or "Usage:" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help" in help_result.output
