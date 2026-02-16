import mlflow 
from mlflow.models import infer_signature

class ExpreimentTracker:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def log_training(self, params, metrics, model, X_test, y_test):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_test, y_test)
            mlflow.keras.log_model(model, "model", signature=signature)