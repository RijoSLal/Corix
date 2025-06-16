import logging
import dagshub
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from sklearn.base import BaseEstimator  # for type hinting sklearn models
from tensorflow import Module as TFModel  # basic TensorFlow model type

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DagsHubMLflowLogger:
    def __init__(self,repo_owner,repo_name) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        """
        Initialize MLflow tracking with DagsHub integration.
        """
        dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
        logger.info("Initialized MLflow tracking with DagsHub")

    def log_sklearn_model(
        self,
        model: BaseEstimator,
        params: dict,
        accuracy: float,
        f1_score: float,
        experiment_name: str = "Scikit-Learn Model Tracking"
    ) -> None:
        """
        Logs a scikit-learn model, its parameters, and performance metrics to MLflow.

        Args:
            model: A trained scikit-learn model.
            params: A dictionary of model hyperparameters.
            accuracy: Accuracy score of the model.
            f1_score: F1 score of the model.
            experiment_name: The MLflow experiment name.
        """
        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="sklearn-run"):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "f1_score": f1_score
                })
                mlflow.sklearn.log_model(model, "model")
                mlflow.end_run()
                logger.info("Scikit-learn model logged successfully.")
        except Exception as e:
            logger.error(f"Error logging Scikit-learn model: {e}")

    def log_tensorflow_model(
        self,
        model: TFModel,
        loss: float,
        acc: float,
        experiment_name: str = "TensorFlow Model Tracking"
    ) -> None:
        """
        Logs a TensorFlow model and its performance metrics to MLflow.

        Args:
            model: A trained TensorFlow model.
            loss: Loss value of the model.
            acc: Accuracy of the model.
            experiment_name: The MLflow experiment name.
        """
        try:

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="tensorflow-run"):
                mlflow.log_metric("loss", loss)
                mlflow.log_metric("accuracy", acc)
                mlflow.tensorflow.log_model(model, "tf_model")
                mlflow.end_run()
                logger.info("TensorFlow model logged successfully.")

        except Exception as e:
            logger.error(f"Error logging TensorFlow model: {e}")