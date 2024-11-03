# causal_analysis_example.py
from sklearn.datasets import load_diabetes
from lazypredict.estimators.causal_analysis import LazyCausalAnalyzer
from lazypredict.metrics.causal_analysis_metrics import CausalAnalysisMetrics
from lazypredict.utils.logger import Logger

# Load dataset
data = load_diabetes(as_frame=True)
X, y = data.data, data.target

# Logger setup
logger = Logger.configure_logger("causal_analysis_example")

# Model Training and Evaluation
model = LazyCausalAnalyzer()
model.fit(X, y)
predictions = model.predict(X)

# Compute Metrics
metrics_calculator = CausalAnalysisMetrics()
metrics = metrics_calculator.compute(y, predictions)
print("Metrics:", metrics)
