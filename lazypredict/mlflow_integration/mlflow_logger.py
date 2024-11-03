import mlflow
import mlflow.sklearn

class MLflowLogger:
    """
    MLflowLogger for tracking model experiments, parameters, metrics, and artifacts using MLflow.

    This class provides methods to log models and their relevant information to MLflow.

    Attributes
    ----------
    experiment_name : str
        Name of the MLflow experiment.
    run : mlflow.ActiveRun
        The active MLflow run, if any.

    Methods
    -------
    start_run(run_name=None):
        Start an MLflow run with the given name.
    log_params(params):
        Log model parameters to MLflow.
    log_metrics(metrics):
        Log model metrics to MLflow.
    log_model(model, model_name):
        Log the model itself to MLflow.
    end_run():
        End the current MLflow run.
    """

    def __init__(self, experiment_name="Default"):
        """
        Parameters
        ----------
        experiment_name : str, optional
            Name of the MLflow experiment. Default is "Default".
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.run = None

    def start_run(self, run_name=None):
        """
        Start an MLflow run.

        Parameters
        ----------
        run_name : str, optional
            Name of the run. If None, MLflow will assign a default name.

        Returns
        -------
        None
        """
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params : dict
            Dictionary of parameters to log.

        Returns
        -------
        None
        """
        if self.run:
            mlflow.log_params(params)
        else:
            raise ValueError("No active MLflow run. Please start a run before logging parameters.")

    def log_metrics(self, metrics):
        """
        Log metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics to log.

        Returns
        -------
        None
        """
        if self.run:
            mlflow.log_metrics(metrics)
        else:
            raise ValueError("No active MLflow run. Please start a run before logging metrics.")

    def log_model(self, model, model_name="model"):
        """
        Log a model to MLflow.

        Parameters
        ----------
        model : object
            Trained model to log.
        model_name : str, optional
            Name to save the model under in MLflow. Default is "model".

        Returns
        -------
        None
        """
        if self.run:
            mlflow.sklearn.log_model(model, model_name)
        else:
            raise ValueError("No active MLflow run. Please start a run before logging the model.")

    def end_run(self):
        """
        End the current MLflow run.

        Returns
        -------
        None
        """
        if self.run:
            mlflow.end_run()
            self.run = None
        else:
            raise ValueError("No active MLflow run to end.")
