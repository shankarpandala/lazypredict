# clustering_example.py
from sklearn.datasets import load_iris
from lazypredict.estimators.clustering import LazyClusterer
from lazypredict.metrics.clustering_metrics import ClusteringMetrics
from lazypredict.utils.logger import Logger

# Load dataset
data = load_iris(as_frame=True)
X = data.data

# Logger setup
logger = Logger.configure_logger("clustering_example")

# Model Training and Evaluation
model = LazyClusterer()
model.fit(X)
labels = model.predict(X)

# Compute Metrics
metrics_calculator = ClusteringMetrics()
metrics = metrics_calculator.compute(X, labels)
print("Metrics:", metrics)
