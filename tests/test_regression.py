import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from lazypredict.regression import LazyRegressor

def test_calculate_metrics():
    # Test case 1: Check if the metrics are calculated correctly
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
    expected_metrics = {
        "R-Squared": 0.9876543209876543,
        "Adjusted R-Squared": 0.9753086419753086,
        "RMSE": 0.4242640687119285
    }
    regressor = LazyRegressor()
    metrics = regressor._calculate_metrics(y_true, y_pred)
    assert metrics == expected_metrics

def test_fit():
    # Test case 2: Check if the results dataframe is generated correctly
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[7, 8], [9, 10]])
    y_train = np.array([1, 2, 3])
    y_test = np.array([4, 5])
    regressor = LazyRegressor()
    results_df = regressor.fit(X_train, X_test, y_train, y_test)
    assert isinstance(results_df, pd.DataFrame)
    assert results_df.shape[0] == len(regressor.estimators)

    # Test case 3: Check if the results dataframe is sorted correctly
    assert results_df["Adjusted R-Squared"].is_monotonic_decreasing

    # Test case 4: Check if the metrics are calculated and added to the results dataframe
    assert "R-Squared" in results_df.columns
    assert "Adjusted R-Squared" in results_df.columns
    assert "RMSE" in results_df.columns

# Additional test cases can be added to cover more scenarios
