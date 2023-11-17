import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from lazypredict.classification import LazyClassifier

class TestLazyClassifier:
    @pytest.fixture
    def lazy_classifier(self):
        return LazyClassifier()

    def test_calculate_metrics(self, lazy_classifier):
        # Define the necessary data
        y_true = [0, 1, 2]
        y_pred = [0, 1, 3]

        # Call the _calculate_metrics method
        metrics = lazy_classifier._calculate_metrics(y_true, y_pred)

        # Assert that the metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert "Accuracy" in metrics
        assert "Balanced Accuracy" in metrics
        assert "F1 Score" in metrics
        assert "ROC AUC" in metrics

    def test_fit(self, lazy_classifier):
        # Define the necessary data
        X_train = [[1, 2], [3, 4]]
        X_test = [[5, 6], [7, 8]]
        y_train = [0, 1]
        y_test = [2, 3]

        # Call the fit method
        results_df = lazy_classifier.fit(X_train, X_test, y_train, y_test)

        # Assert that the results dataframe is created correctly
        assert isinstance(results_df, pd.DataFrame)
        assert "Model" in results_df.columns
        assert "Accuracy" in results_df.columns
        assert "Balanced Accuracy" in results_df.columns
        assert "F1 Score" in results_df.columns
        assert "ROC AUC" in results_df.columns
        assert len(results_df) > 0
        assert results_df["Balanced Accuracy"].isnull().sum() == 0
        assert results_df["ROC AUC"].isnull().sum() <= len(results_df) - 1
        assert results_df["Balanced Accuracy"].is_monotonic_decreasing
