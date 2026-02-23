from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow wrapper with graceful degradation when MLflow is unavailable."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self._enabled = False
        self._mlflow = None
        self._infer_signature = None

        try:
            import mlflow
            from mlflow.models import infer_signature

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._infer_signature = infer_signature
            self._enabled = True
        except Exception as exc:
            logger.warning("MLflow disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def _run(self):
        if not self._enabled:
            yield None
            return

        with self._mlflow.start_run() as run:
            yield run

    def log_training(
        self,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model: Any = None,
        X_test: Any = None,
        y_test: Any = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a training event and optional model artifact to MLflow."""
        if not self._enabled:
            logger.info("MLflow tracking skipped (disabled)")
            return

        with self._run():
            if params:
                safe_params = {k: str(v) for k, v in params.items()}
                self._mlflow.log_params(safe_params)

            if metrics:
                numeric_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                if numeric_metrics:
                    self._mlflow.log_metrics(numeric_metrics)

            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    self._mlflow.log_artifact(artifact_path, artifact_name)

            if model is not None:
                signature = None
                if X_test is not None and y_test is not None:
                    try:
                        signature = self._infer_signature(X_test, y_test)
                    except Exception as exc:
                        logger.warning("Failed to infer model signature: %s", exc)

                self._mlflow.keras.log_model(model, "model", signature=signature)

    def compare_experiments(self, order_by: str = "metrics.rmse ASC", max_results: int = 10):
        if not self._enabled:
            return []

        return self._mlflow.search_runs(
            experiment_names=[self.experiment_name],
            order_by=[order_by],
            max_results=max_results,
        )
