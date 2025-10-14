import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from click.testing import CliRunner
from lazypredict import cli


@pytest.fixture
def classification_csv(tmp_path):
    """Create a temporary classification CSV file."""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    csv_path = tmp_path / "classification_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def regression_csv(tmp_path):
    """Create a temporary regression CSV file."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    csv_path = tmp_path / "regression_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def invalid_csv(tmp_path):
    """Create a temporary invalid CSV file."""
    csv_path = tmp_path / "invalid_data.csv"
    with open(csv_path, 'w') as f:
        f.write("This is not a valid CSV file\n")
        f.write("Random text here\n")
    return str(csv_path)


class TestCLIMain:
    """Test main CLI functionality."""

    def test_cli_main(self):
        """Test main CLI entry point."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0

    def test_cli_help(self):
        """Test help message."""
        runner = CliRunner()
        result = runner.invoke(cli.main, ["--help"])
        assert result.exit_code == 0
        assert "LazyPredict" in result.output
        assert "classify" in result.output
        assert "regress" in result.output
        assert "info" in result.output

    def test_cli_version(self):
        """Test version option."""
        runner = CliRunner()
        result = runner.invoke(cli.main, ["--version"])
        assert result.exit_code == 0
        assert "0.2.16" in result.output


class TestClassifyCommand:
    """Test classify command."""

    def test_classify_basic(self, classification_csv):
        """Test basic classification command."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0
        assert "Loading data" in result.output
        assert "Training models" in result.output
        assert "CLASSIFICATION RESULTS" in result.output

    def test_classify_with_output(self, classification_csv, tmp_path):
        """Test classification with output file."""
        output_file = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--output', str(output_file),
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        assert "Results saved" in result.output

        # Verify output file content
        results_df = pd.read_csv(output_file)
        assert len(results_df) > 0
        assert 'Accuracy' in results_df.columns

    def test_classify_with_predictions(self, classification_csv, tmp_path):
        """Test classification with predictions output."""
        output_file = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--output', str(output_file),
            '--predictions',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0
        assert "Predictions saved" in result.output

        # Verify predictions file exists
        pred_file = tmp_path / "results_predictions.csv"
        assert os.path.exists(pred_file)

    def test_classify_missing_target(self, classification_csv):
        """Test classification with missing target column."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'nonexistent_column'
        ])
        assert result.exit_code != 0
        assert "Target column 'nonexistent_column' not found" in result.output

    def test_classify_missing_file(self):
        """Test classification with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            'nonexistent_file.csv',
            '--target', 'target'
        ])
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Error" in result.output

    def test_classify_test_size(self, classification_csv):
        """Test classification with custom test size."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--test-size', '0.2',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0

    def test_classify_random_state(self, classification_csv):
        """Test classification with custom random state."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--random-state', '123',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0

    def test_classify_verbose(self, classification_csv):
        """Test classification with verbose output."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--verbose',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0

    def test_classify_parallel(self, classification_csv):
        """Test classification with parallel processing."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'classify',
            classification_csv,
            '--target', 'target',
            '--n-jobs', '-1'
        ])
        assert result.exit_code == 0


class TestRegressCommand:
    """Test regress command."""

    def test_regress_basic(self, regression_csv):
        """Test basic regression command."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'regress',
            regression_csv,
            '--target', 'target',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0
        assert "Loading data" in result.output
        assert "Training models" in result.output
        assert "REGRESSION RESULTS" in result.output

    def test_regress_with_output(self, regression_csv, tmp_path):
        """Test regression with output file."""
        output_file = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'regress',
            regression_csv,
            '--target', 'target',
            '--output', str(output_file),
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        assert "Results saved" in result.output

        # Verify output file content
        results_df = pd.read_csv(output_file)
        assert len(results_df) > 0
        assert 'R-Squared' in results_df.columns

    def test_regress_with_predictions(self, regression_csv, tmp_path):
        """Test regression with predictions output."""
        output_file = tmp_path / "results.csv"
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'regress',
            regression_csv,
            '--target', 'target',
            '--output', str(output_file),
            '--predictions',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0
        assert "Predictions saved" in result.output

        # Verify predictions file exists
        pred_file = tmp_path / "results_predictions.csv"
        assert os.path.exists(pred_file)

    def test_regress_missing_target(self, regression_csv):
        """Test regression with missing target column."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'regress',
            regression_csv,
            '--target', 'nonexistent_column'
        ])
        assert result.exit_code != 0
        assert "Target column 'nonexistent_column' not found" in result.output

    def test_regress_test_size(self, regression_csv):
        """Test regression with custom test size."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'regress',
            regression_csv,
            '--target', 'target',
            '--test-size', '0.2',
            '--n-jobs', '1'
        ])
        assert result.exit_code == 0

    def test_regress_parallel(self, regression_csv):
        """Test regression with parallel processing."""
        runner = CliRunner()
        result = runner.invoke(cli.main, [
            'regress',
            regression_csv,
            '--target', 'target',
            '--n-jobs', '-1'
        ])
        assert result.exit_code == 0


class TestInfoCommand:
    """Test info command."""

    def test_info_command(self):
        """Test info command output."""
        runner = CliRunner()
        result = runner.invoke(cli.main, ['info'])
        assert result.exit_code == 0
        assert "LazyPredict" in result.output
        assert "Version: 0.2.16" in result.output
        assert "Automated model selection" in result.output
        assert "40+ classification and regression models" in result.output

    def test_info_help(self):
        """Test info command help."""
        runner = CliRunner()
        result = runner.invoke(cli.main, ['info', '--help'])
        assert result.exit_code == 0
        assert "Display package information" in result.output