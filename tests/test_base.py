import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lazypredict.base import LazyBaseEstimator

class TestLazyBaseEstimator:
    @pytest.fixture
    def lazy_estimator(self):
        return LazyBaseEstimator()

    def test_create_pipeline(self, lazy_estimator):
        # Define the necessary transformers and features
        numeric_transformer = "numeric_transformer"
        categorical_transformer_low = "categorical_transformer_low"
        categorical_transformer_high = "categorical_transformer_high"
        numeric_features = ["feature1", "feature2"]
        categorical_low = ["category1"]
        categorical_high = ["category2"]

        # Call the _create_pipeline method
        pipeline = lazy_estimator._create_pipeline("estimator")

        # Assert that the pipeline is created correctly
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "preprocessor"
        assert isinstance(pipeline.steps[0][1], ColumnTransformer)
        assert len(pipeline.steps[0][1].transformers) == 3
        assert pipeline.steps[0][1].transformers[0][0] == "numeric"
        assert pipeline.steps[0][1].transformers[0][1] == numeric_transformer
        assert pipeline.steps[0][1].transformers[0][2] == numeric_features
        assert pipeline.steps[0][1].transformers[1][0] == "categorical_low"
        assert pipeline.steps[0][1].transformers[1][1] == categorical_transformer_low
        assert pipeline.steps[0][1].transformers[1][2] == categorical_low
        assert pipeline.steps[0][1].transformers[2][0] == "categorical_high"
        assert pipeline.steps[0][1].transformers[2][1] == categorical_transformer_high
        assert pipeline.steps[0][1].transformers[2][2] == categorical_high
        assert pipeline.steps[1][0] == "estimator"
        assert pipeline.steps[1][1] == "estimator"

    def test_fit_estimator(self, lazy_estimator):
        # Define the necessary data and estimator
        X_train = [[1, 2], [3, 4]]
        y_train = [0, 1]
        X_test = [[5, 6], [7, 8]]
        y_test = [2, 3]
        estimator = "estimator"

        # Call the _fit_estimator method
        metrics = lazy_estimator._fit_estimator(X_train, y_train, X_test, y_test, estimator)

        # Assert that the metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert "Time Taken" in metrics

    def test_calculate_metrics(self, lazy_estimator):
        # Define the necessary data
        y_true = [0, 1, 2]
        y_pred = [0, 1, 3]

        # Call the _calculate_metrics method
        metrics = lazy_estimator._calculate_metrics(y_true, y_pred)

        # Assert that the metrics are calculated correctly
        assert isinstance(metrics, dict)
        assert "Time Taken" not in metrics

    def test_fit(self, lazy_estimator):
        # Define the necessary data
        X_train = [[1, 2], [3, 4]]
        X_test = [[5, 6], [7, 8]]
        y_train = [0, 1]
        y_test = [2, 3]

        # Call the fit method
        lazy_estimator.fit(X_train, X_test, y_train, y_test)

        # Assert that the models dictionary is updated
        assert len(lazy_estimator.models) == 1
