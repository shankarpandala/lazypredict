import pytest
from click.testing import CliRunner
from lazypredict import cli

def test_cli_main():
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "lazypredict.cli.main" in result.output

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    assert "--help  Show this message and exit." in result.output