import pytest
from click.testing import CliRunner
from sklearn.datasets import load_iris

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


@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True).frame
    path = tmp_path / "iris.csv"
    data.to_csv(path, index=False)
    return str(path)


def test_cli_unknown_target_column_fails(iris_csv):
    result = CliRunner().invoke(
        cli.main, ["--task", "classification", "--input", iris_csv, "--target", "nope"]
    )
    assert result.exit_code == 2
    assert "'nope' not found" in result.output


@pytest.mark.parametrize(
    "args, missing",
    [
        (["--task", "classification"], "--input, --target"),
        (["--target", "target"], "--task, --input"),
    ],
)
def test_cli_partial_options_fail(args, missing):
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 2
    assert missing in result.output


def test_cli_rejects_invalid_test_size(iris_csv):
    result = CliRunner().invoke(
        cli.main,
        ["--task", "classification", "--input", iris_csv, "--target", "target", "--test-size", "1.5"],
    )
    assert result.exit_code == 2
    assert "--test-size" in result.output


def test_cli_classification_end_to_end(iris_csv):
    result = CliRunner().invoke(
        cli.main, ["--task", "classification", "--input", iris_csv, "--target", "target"]
    )
    assert result.exit_code == 0, result.output
    assert "Balanced Accuracy" in result.output
    assert "LogisticRegression" in result.output


def _write_csv(tmp_path, frame):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_cli_fails_when_no_model_can_be_trained(tmp_path):
    # Issue #629: an empty table was printed with exit status 0.
    X, _ = load_iris(return_X_y=True, as_frame=True)
    frame = X.assign(label=["high", "low", "mid"] * 50)
    args = ["--task", "regression", "--input", _write_csv(tmp_path, frame), "--target", "label"]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 1
    assert "No model could be trained" in result.output


def test_cli_random_state_seeds_the_models(tmp_path, monkeypatch):
    import lazypredict.Supervised as supervised

    seeds = []

    class Recording(supervised.LazyClassifier):
        def __init__(self, **kwargs):
            seeds.append(kwargs.get("random_state"))
            super().__init__(**{**kwargs, "classifiers": ["DecisionTreeClassifier"]})

    monkeypatch.setattr(supervised, "LazyClassifier", Recording)
    X, y = load_iris(return_X_y=True, as_frame=True)
    path = _write_csv(tmp_path, X.assign(label=y))
    args = ["--task", "classification", "--input", path, "--target", "label", "--random-state", "7"]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    assert seeds == [7]
