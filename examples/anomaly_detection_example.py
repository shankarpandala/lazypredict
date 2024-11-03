# anomaly_detection_example.py
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from lazypredict.estimators.anomaly_detection import LazyAnomalyDetector
from lazypredict.metrics.anomaly_detection_metrics import AnomalyDetectionMetrics
from lazypredict.utils.logger import Logger

# Load dataset
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

# Standardize data for anomaly detection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logger setup
logger = Logger.configure_logger("anomaly_detection_example")

# Model Training and Evaluation
model = LazyAnomalyDetector()
model.fit(X_scaled)
predictions = model.predict(X_scaled)

# Compute Metrics
metrics_calculator = AnomalyDetectionMetrics()
metrics = metrics_calculator.compute(y, predictions)
print("Metrics:", metrics)
