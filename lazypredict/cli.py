# -*- coding: utf-8 -*-

"""Console script for lazypredict — quick model benchmarking from the command line."""
import sys

import click

from lazypredict import __version__


@click.command()
@click.version_option(version=__version__, prog_name="lazypredict")
@click.option(
    "--task",
    type=click.Choice(["classification", "regression"]),
    help="Type of ML task.",
)
@click.option("--input", "input_file", type=click.Path(exists=True), help="Path to CSV dataset.")
@click.option("--target", type=str, help="Name of the target column.")
@click.option("--test-size", type=float, default=0.2, help="Fraction of data for testing.")
@click.option("--random-state", type=int, default=42, help="Random seed.")
def main(task, input_file, target, test_size, random_state):
    """LazyPredict — quickly benchmark scikit-learn models on a dataset."""
    if input_file is None or target is None or task is None:
        click.echo("LazyPredict CLI — run `lazypredict --help` for usage.")
        click.echo("Example: lazypredict --task classification --input data.csv --target label")
        return 0

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_file)
    if target not in df.columns:
        click.echo(f"Error: target column '{target}' not found in {input_file}", err=True)
        return 1

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if task == "classification":
        from lazypredict.Supervised import LazyClassifier
        model = LazyClassifier(verbose=0, ignore_warnings=True)
    else:
        from lazypredict.Supervised import LazyRegressor
        model = LazyRegressor(verbose=0, ignore_warnings=True)

    scores, _ = model.fit(X_train, X_test, y_train, y_test)
    click.echo(scores.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
