import pytest
from click.testing import CliRunner
from lazypredict import cli
from lazypredict.cli import main
import unittest

class TestCLI(unittest.TestCase):
    def test_main(self):
        runner = CliRunner()
        result = runner.invoke(cli.main)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("lazypredict.cli.main", result.output)

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli.main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--help  Show this message and exit.", result.output)

if __name__ == '__main__':
    unittest.main()