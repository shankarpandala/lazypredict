# examples/causal_analysis_example.py

"""
Causal Analysis Example using LazyCausalAnalyzer from lazypredict.

This script demonstrates how to use LazyCausalAnalyzer to perform causal analysis
on a synthetic dataset.
"""

from lazypredict.utils.backend import Backend

DataFrame = Backend.DataFrame
Series = Backend.Series
import numpy as np

from lazypredict.estimators import LazyCausalAnalyzer
from lazypredict.metrics import CausalAnalysisMetrics
from lazypredict.utils.backend import Backend

# Initialize the backend (pandas is default)
Backend.initialize_backend(use_gpu=False)

def main():
    # Create a synthetic dataset
    np.random.seed(42)
    size = 1000
    treatment = np.random.binomial(1, 0.5, size)
    confounder = np.random.normal(0, 1, size)
    outcome = 2 * treatment + 3 * confounder + np.random.normal(0, 1, size)

    data = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome,
        'confounder': confounder,
    })

    # Initialize LazyCausalAnalyzer
    analyzer = LazyCausalAnalyzer(
        verbose=1,
        ignore_warnings=False,
        random_state=42,
        use_gpu=False,
        mlflow_logging=False,
        treatment='treatment',
        outcome='outcome',
        common_causes=['confounder'],
    )

    # Fit models and get results
    results = analyzer.fit(data)

    # Display results
    print("Causal Analysis Model Evaluation Results:")
    print(results)

    # Access trained models
    models = analyzer.models

if __name__ == "__main__":
    main()
