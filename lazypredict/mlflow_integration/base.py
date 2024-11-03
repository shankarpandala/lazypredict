    def __del__(self):
        """
        Ensures that the MLflow run is ended when the estimator is destroyed.
        """
        if self.mlflow_logging and self.mlflow_logger:
            self.mlflow_logger.end_run()
