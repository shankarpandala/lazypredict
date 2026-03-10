from click.testing import CliRunner
from lazypredict import cli


def test_cli_main():
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "LazyPredict CLI" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    assert "--help" in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0
