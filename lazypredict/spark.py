"""Spark MLlib integration for LazyPredict.

Provides LazySparkClassifier and LazySparkRegressor that wrap
PySpark MLlib estimators for large-scale distributed model comparison.

Requires: pip install lazypredict[spark]
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("lazypredict")

# Optional PySpark
try:
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.classification import (
        DecisionTreeClassifier as SparkDTC,
        GBTClassifier as SparkGBTC,
        LinearSVC as SparkLinearSVC,
        LogisticRegression as SparkLR,
        NaiveBayes as SparkNB,
        RandomForestClassifier as SparkRFC,
    )
    from pyspark.ml.evaluation import (
        MulticlassClassificationEvaluator,
        RegressionEvaluator,
    )
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import (
        DecisionTreeRegressor as SparkDTR,
        GBTRegressor as SparkGBTR,
        GeneralizedLinearRegression as SparkGLR,
        LinearRegression as SparkLinReg,
        RandomForestRegressor as SparkRFR,
    )
    from pyspark.sql import DataFrame as SparkDataFrame

    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False


def _check_spark():
    if not _SPARK_AVAILABLE:
        raise ImportError(
            "PySpark is required for Spark MLlib models. "
            "Install with: pip install lazypredict[spark]"
        )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_SPARK_CLASSIFIERS = {}
_SPARK_REGRESSORS = {}

if _SPARK_AVAILABLE:
    _SPARK_CLASSIFIERS = {
        "SparkLogisticRegression": SparkLR,
        "SparkDecisionTreeClassifier": SparkDTC,
        "SparkRandomForestClassifier": SparkRFC,
        "SparkGBTClassifier": SparkGBTC,
        "SparkNaiveBayes": SparkNB,
        "SparkLinearSVC": SparkLinearSVC,
    }
    _SPARK_REGRESSORS = {
        "SparkLinearRegression": SparkLinReg,
        "SparkDecisionTreeRegressor": SparkDTR,
        "SparkRandomForestRegressor": SparkRFR,
        "SparkGBTRegressor": SparkGBTR,
        "SparkGeneralizedLinearRegression": SparkGLR,
    }


class LazySparkClassifier:
    """Fit all Spark MLlib classification models and benchmark them.

    Parameters
    ----------
    verbose : int, optional (default=0)
        Verbosity level.
    ignore_warnings : bool, optional (default=True)
        Suppress model errors.
    label_col : str, optional (default="label")
        Name of the label column.
    features_col : str, optional (default="features")
        Name for the assembled features column.
    max_models : int or None, optional (default=None)
        Maximum number of models to train.

    Requires PySpark (``pip install lazypredict[spark]``).
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        label_col: str = "label",
        features_col: str = "features",
        max_models: Optional[int] = None,
    ):
        _check_spark()
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.label_col = label_col
        self.features_col = features_col
        self.max_models = max_models
        self.models: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}

    def fit(
        self,
        train_df: "SparkDataFrame",
        test_df: "SparkDataFrame",
        feature_cols: Optional[List[str]] = None,
    ):
        """Fit Spark MLlib classifiers and evaluate.

        Parameters
        ----------
        train_df : pyspark.sql.DataFrame
            Training data.
        test_df : pyspark.sql.DataFrame
            Test data.
        feature_cols : list of str or None
            Feature column names. If None, all columns except label_col.

        Returns
        -------
        pd.DataFrame
            Metrics for every model.
        """
        import pandas as pd

        if feature_cols is None:
            feature_cols = [c for c in train_df.columns if c != self.label_col]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=self.features_col,
            handleInvalid="skip",
        )

        estimators = list(_SPARK_CLASSIFIERS.items())
        if self.max_models:
            estimators = estimators[:self.max_models]

        results = []
        for name, cls in estimators:
            start = time.time()
            try:
                model = cls(
                    labelCol=self.label_col,
                    featuresCol=self.features_col,
                )
                pipe = SparkPipeline(stages=[assembler, model])
                fitted = pipe.fit(train_df)
                predictions = fitted.transform(test_df)

                evaluator = MulticlassClassificationEvaluator(
                    labelCol=self.label_col
                )
                accuracy = evaluator.evaluate(
                    predictions, {evaluator.metricName: "accuracy"}
                )
                f1 = evaluator.evaluate(
                    predictions, {evaluator.metricName: "f1"}
                )
                precision = evaluator.evaluate(
                    predictions, {evaluator.metricName: "weightedPrecision"}
                )
                recall = evaluator.evaluate(
                    predictions, {evaluator.metricName: "weightedRecall"}
                )

                self.models[name] = fitted
                elapsed = time.time() - start
                results.append({
                    "Model": name,
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall,
                    "Time Taken": elapsed,
                })

                if self.verbose > 0:
                    logger.info(
                        "Model=%s Accuracy=%.4f F1=%.4f Time=%.2fs",
                        name, accuracy, f1, elapsed,
                    )

            except Exception as exc:
                self.errors[name] = exc
                if not self.ignore_warnings:
                    logger.warning("%s failed: %s", name, exc)

        scores = pd.DataFrame(results)
        if not scores.empty:
            scores = scores.sort_values("Accuracy", ascending=False).set_index("Model")
        return scores


class LazySparkRegressor:
    """Fit all Spark MLlib regression models and benchmark them.

    Parameters
    ----------
    verbose : int, optional (default=0)
        Verbosity level.
    ignore_warnings : bool, optional (default=True)
        Suppress model errors.
    label_col : str, optional (default="label")
        Name of the label column.
    features_col : str, optional (default="features")
        Name for the assembled features column.
    max_models : int or None, optional (default=None)
        Maximum number of models to train.

    Requires PySpark (``pip install lazypredict[spark]``).
    """

    def __init__(
        self,
        verbose: int = 0,
        ignore_warnings: bool = True,
        label_col: str = "label",
        features_col: str = "features",
        max_models: Optional[int] = None,
    ):
        _check_spark()
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.label_col = label_col
        self.features_col = features_col
        self.max_models = max_models
        self.models: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}

    def fit(
        self,
        train_df: "SparkDataFrame",
        test_df: "SparkDataFrame",
        feature_cols: Optional[List[str]] = None,
    ):
        """Fit Spark MLlib regressors and evaluate.

        Parameters
        ----------
        train_df : pyspark.sql.DataFrame
            Training data.
        test_df : pyspark.sql.DataFrame
            Test data.
        feature_cols : list of str or None
            Feature column names.

        Returns
        -------
        pd.DataFrame
            Metrics for every model.
        """
        import pandas as pd

        if feature_cols is None:
            feature_cols = [c for c in train_df.columns if c != self.label_col]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol=self.features_col,
            handleInvalid="skip",
        )

        estimators = list(_SPARK_REGRESSORS.items())
        if self.max_models:
            estimators = estimators[:self.max_models]

        results = []
        for name, cls in estimators:
            start = time.time()
            try:
                model = cls(
                    labelCol=self.label_col,
                    featuresCol=self.features_col,
                )
                pipe = SparkPipeline(stages=[assembler, model])
                fitted = pipe.fit(train_df)
                predictions = fitted.transform(test_df)

                evaluator = RegressionEvaluator(labelCol=self.label_col)
                rmse = evaluator.evaluate(
                    predictions, {evaluator.metricName: "rmse"}
                )
                r2 = evaluator.evaluate(
                    predictions, {evaluator.metricName: "r2"}
                )
                mae = evaluator.evaluate(
                    predictions, {evaluator.metricName: "mae"}
                )

                self.models[name] = fitted
                elapsed = time.time() - start
                results.append({
                    "Model": name,
                    "R-Squared": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "Time Taken": elapsed,
                })

                if self.verbose > 0:
                    logger.info(
                        "Model=%s R2=%.4f RMSE=%.4f Time=%.2fs",
                        name, r2, rmse, elapsed,
                    )

            except Exception as exc:
                self.errors[name] = exc
                if not self.ignore_warnings:
                    logger.warning("%s failed: %s", name, exc)

        scores = pd.DataFrame(results)
        if not scores.empty:
            scores = scores.sort_values("R-Squared", ascending=False).set_index("Model")
        return scores
