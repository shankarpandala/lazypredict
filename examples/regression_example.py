# regression_example.py
from sklearn.datasets import fetch_california_housing
from lazypredict.estimators.regression import LazyRegressor
from lazypredict.preprocessing.feature_engineering import FeatureEngineering
from lazypredict.preprocessing.feature_selection import FeatureSelection
from lazypredict.mlflow_integration import MLflowLogger
from lazypredict.metrics.regression_metrics import RegressionMetrics
from lazypredict.utils.logger import Logger

# Load dataset
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

# Logger setup
logger = Logger.configure_logger("regression_example")

# Initialize MLflow logger
mlflow_logger = MLflowLogger(experiment_name="Regression Experiment")
mlflow_logger.start_run(run_name="California Housing Regression")

# Feature Engineering (Polynomial Features)
feature_engineer = FeatureEngineering(method="polynomial", degree=2)
X_poly = feature_engineer.transform(X)

# Feature Selection
feature_selector = FeatureSelection(k=5)
feature_selector.fit(X_poly, y)
X_selected = feature_selector.transform(X_poly)

# Model Training and Evaluation
model = LazyRegressor()
model.fit(X_selected, y)
predictions = model.predict(X_selected)

# Compute Metrics
metrics_calculator = RegressionMetrics()
metrics = metrics_calculator.compute(y, predictions)
print("Metrics:", metrics)

# Log parameters and metrics to MLflow
mlflow_logger.log_params({"feature_engineering": "polynomial", "feature_selection_k": 5})
mlflow_logger.log_metrics(metrics)
mlflow_logger.log_model(model.model, model_name="LazyRegressor")

# End MLflow run
mlflow_logger.end_run()

# SHAP Explainability (for Jupyter Notebook)
# Uncomment the lines below to use in a notebook
# from lazypredict.explainability.shap_explainer import SHAPExplainer
# shap_explainer = SHAPExplainer(model.model)
# shap_values = shap_explainer.explain(X_selected)
# shap_explainer.plot_summary(shap_values, X_selected)

# LIME Explainability (for Jupyter Notebook)
# Uncomment the lines below to use in a notebook
# from lazypredict.explainability.lime_explainer import LIMEExplainer
# lime_explainer = LIMEExplainer(model.model, X_selected)
# explanation = lime_explainer.explain_instance(X_selected, instance=0)
# print("LIME Explanation:", explanation)
