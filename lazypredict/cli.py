# -*- coding: utf-8 -*-

"""Console script for lazypredict."""
import sys
import os
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor


@click.group()
@click.version_option(version='0.2.16')
def main():
    """
    LazyPredict - Automated Machine Learning Model Selection
    
    Train and compare multiple ML models with minimal code.
    """
    pass


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', default=0.3, help='Test set size (default: 0.3)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--predictions/--no-predictions', default=False, help='Save predictions')
@click.option('--n-jobs', default=1, help='Number of parallel jobs (-1 for all CPUs)')
@click.option('--random-state', default=42, help='Random state for reproducibility')
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
def classify(input_file, target, test_size, output, predictions, n_jobs, random_state, verbose):
    """
    Run classification on a dataset.

    Example:
        lazypredict classify data.csv --target species --n-jobs -1
    """
    try:
        # Load data
        click.echo(f"Loading data from {input_file}...")
        try:
            df = pd.read_csv(input_file)
        except pd.errors.EmptyDataError:
            click.echo("Error: The CSV file is empty", err=True)
            return 1
        except pd.errors.ParserError as e:
            click.echo(f"Error: Failed to parse CSV file - {str(e)}", err=True)
            return 1
        except Exception as e:
            click.echo(f"Error: Failed to read CSV file - {str(e)}", err=True)
            return 1

        if target not in df.columns:
            click.echo(f"Error: Target column '{target}' not found in dataset", err=True)
            click.echo(f"Available columns: {', '.join(df.columns)}", err=True)
            return 1
        
        # Split features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        click.echo(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        click.echo(f"Training set: {X_train.shape[0]} samples")
        click.echo(f"Test set: {X_test.shape[0]} samples")
        click.echo(f"Target classes: {y.nunique()}")
        click.echo("\nTraining models...")
        
        # Train models
        clf = LazyClassifier(
            verbose=verbose,
            ignore_warnings=True,
            predictions=predictions,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        if predictions:
            models, preds = clf.fit(X_train, X_test, y_train, y_test)
        else:
            models = clf.fit(X_train, X_test, y_train, y_test)
        
        # Display results
        click.echo("\n" + "="*80)
        click.echo("CLASSIFICATION RESULTS")
        click.echo("="*80)
        click.echo(models.to_string())

        # Save results
        if output:
            # Validate output directory exists
            output_dir = os.path.dirname(output)
            if output_dir and not os.path.exists(output_dir):
                click.echo(f"Error: Output directory does not exist: {output_dir}", err=True)
                return 1

            models.to_csv(output)
            click.echo(f"\nResults saved to {output}")

            if predictions:
                pred_file = output.replace('.csv', '_predictions.csv')
                preds.to_csv(pred_file, index=False)
                click.echo(f"Predictions saved to {pred_file}")

        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--test-size', default=0.3, help='Test set size (default: 0.3)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--predictions/--no-predictions', default=False, help='Save predictions')
@click.option('--n-jobs', default=1, help='Number of parallel jobs (-1 for all CPUs)')
@click.option('--random-state', default=42, help='Random state for reproducibility')
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
def regress(input_file, target, test_size, output, predictions, n_jobs, random_state, verbose):
    """
    Run regression on a dataset.

    Example:
        lazypredict regress data.csv --target price --n-jobs -1
    """
    try:
        # Load data
        click.echo(f"Loading data from {input_file}...")
        try:
            df = pd.read_csv(input_file)
        except pd.errors.EmptyDataError:
            click.echo("Error: The CSV file is empty", err=True)
            return 1
        except pd.errors.ParserError as e:
            click.echo(f"Error: Failed to parse CSV file - {str(e)}", err=True)
            return 1
        except Exception as e:
            click.echo(f"Error: Failed to read CSV file - {str(e)}", err=True)
            return 1

        if target not in df.columns:
            click.echo(f"Error: Target column '{target}' not found in dataset", err=True)
            click.echo(f"Available columns: {', '.join(df.columns)}", err=True)
            return 1
        
        # Split features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        click.echo(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        click.echo(f"Training set: {X_train.shape[0]} samples")
        click.echo(f"Test set: {X_test.shape[0]} samples")
        click.echo(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        click.echo("\nTraining models...")
        
        # Train models
        reg = LazyRegressor(
            verbose=verbose,
            ignore_warnings=True,
            predictions=predictions,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        if predictions:
            models, preds = reg.fit(X_train, X_test, y_train, y_test)
        else:
            models = reg.fit(X_train, X_test, y_train, y_test)
        
        # Display results
        click.echo("\n" + "="*80)
        click.echo("REGRESSION RESULTS")
        click.echo("="*80)
        click.echo(models.to_string())

        # Save results
        if output:
            # Validate output directory exists
            output_dir = os.path.dirname(output)
            if output_dir and not os.path.exists(output_dir):
                click.echo(f"Error: Output directory does not exist: {output_dir}", err=True)
                return 1

            models.to_csv(output)
            click.echo(f"\nResults saved to {output}")

            if predictions:
                pred_file = output.replace('.csv', '_predictions.csv')
                preds.to_csv(pred_file, index=False)
                click.echo(f"Predictions saved to {pred_file}")

        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


@main.command()
def info():
    """Display package information."""
    click.echo("LazyPredict - Automated Machine Learning")
    click.echo("Version: 0.2.16")
    click.echo("\nFeatures:")
    click.echo("  - Automated model selection")
    click.echo("  - 40+ classification and regression models")
    click.echo("  - Parallel training support")
    click.echo("  - MLflow integration")
    click.echo("  - Custom metrics")
    click.echo("\nUsage:")
    click.echo("  lazypredict classify <file> --target <column>")
    click.echo("  lazypredict regress <file> --target <column>")
    click.echo("\nFor more information: https://lazypredict.readthedocs.io")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
