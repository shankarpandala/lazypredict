# examples/time_series_example.py

"""
Time Series Forecasting Example using LazyTimeSeriesForecaster from lazypredict.

This script demonstrates how to use LazyTimeSeriesForecaster to automatically fit and evaluate
multiple time series forecasting models on the AirPassengers dataset.
"""

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
from sklearn.model_selection import train_test_split

from lazypredict.estimators import LazyTimeSeriesForecaster
from lazypredict.metrics import TimeSeriesMetrics
from lazypredict.utils.backend import Backend

# Initialize the backend (pandas is default)
Backend.initialize_backend(use_gpu=False)

def main():
    # Load the AirPassengers dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

    # Prepare the data
    df.rename(columns={'Passengers': 'y'}, inplace=True)
    df['ds'] = df.index

    # Split the dataset into training and test sets
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # Initialize LazyTimeSeriesForecaster
    forecaster = LazyTimeSeriesForecaster(
        verbose=1,
        ignore_warnings=False,
        random_state=42,
        use_gpu=False,
        mlflow_logging=False,
        forecast_horizon=len(df_test),
        time_col='ds',
        target_col='y',
    )

    # Fit models and get results
    results, predictions = forecaster.fit(df_train, df_test)

    # Display results
    print("Time Series Forecasting Model Evaluation Results:")
    print(results)

    # Access trained models
    models = forecaster.models

if __name__ == "__main__":
    main()
