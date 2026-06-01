from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import schedule

from src.mlops.alerting import Alerting
from src.mlops.drift_detector import DriftDetector
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.metrics_exporter import MetricsExporter
from src.mlops.audit_store import AuditStore
from src.mlops.model_registry import ModelRegistry
from src.mlops.model_validator import ModelValidator
from src.train_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class AutoRetrainingPipeline:
    """End-to-end retraining with drift checks, validation, and model promotion."""

    def __init__(self, config):
        self.config = config
        self.validator = ModelValidator()
        self.registry = ModelRegistry()
        self.drift_detector = DriftDetector()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "gold_price_retraining")
        self.tracker = ExperimentTracker(experiment_name, tracking_uri=tracking_uri)
        self.alerting = Alerting(service_name="gold-retraining")
        self.metrics_exporter = MetricsExporter()
        self.audit_store = AuditStore(db_path=os.getenv("AUDIT_DB_PATH", "data/mlops_audit.db"))
        self.baseline_metrics = self._load_baseline_metrics()

    def _load_baseline_metrics(self) -> Dict[str, float]:
        production_model = self.registry.get_current_production_model()
        if production_model and production_model.get("metrics"):
            return {
                "rmse": float(production_model["metrics"].get("rmse", 47500.0)),
                "mae": float(production_model["metrics"].get("mae", 32000.0)),
                "r2": float(production_model["metrics"].get("r2", 0.85)),
                "mape": float(production_model["metrics"].get("mape", 15.0)),
            }

        return {"rmse": 47500.0, "mae": 32000.0, "r2": 0.85, "mape": 15.0}

    def trigger_retraining(self) -> bool:
        """Run one retraining cycle and promote if quality gates pass."""
        try:
            logger.info("%s", "=" * 72)
            logger.info("Starting automated retraining pipeline")
            logger.info("%s", "=" * 72)

            train_result = self._train_and_evaluate()
            new_model = train_result["model"]
            new_metrics = train_result["metrics"]
            predicted_prices = train_result["predicted_prices"]
            data = train_result["data"]
            history = train_result["history"]

            if self.drift_detector.reference_data is None:
                self.drift_detector.set_reference_data(data["X_train"])

            drift_report = self.drift_detector.check_drift(data["X_test"])
            drift_detected = bool(drift_report.get("drift_detected", False))
            if drift_detected:
                logger.warning("Data drift detected: %s", drift_report)
                self.metrics_exporter.record_drift()

            valid, validation_report = self.validator.full_validation(
                model=new_model,
                metrics=new_metrics,
                predictions=predicted_prices,
                reference_prices=data.get("metadata", {}).get("test_prices"),
            )

            if not valid:
                logger.error("Model validation failed: %s", validation_report)
                self._log_run(new_metrics, status="rejected", reason="validation_failed")
                self.alerting.send(
                    title="Validation failed",
                    message=str(validation_report),
                    level="warning",
                    alert_key="validation_failed",
                )
                return False

            if drift_detected:
                self._log_run(new_metrics, status="rejected", reason="drift_detected")
                logger.info("Model rejected due to drift gate")
                self.alerting.send(
                    title="Drift detected",
                    message=f"drift_score={drift_report.get('drift_score')}",
                    level="warning",
                    alert_key="drift_detected",
                )
                return False

            if self._should_promote(new_metrics, self.baseline_metrics):
                promote_via_canary = os.getenv(
                    "PROMOTE_VIA_CANARY", "true"
                ).lower() in {"1", "true", "yes"}
                if promote_via_canary:
                    registration = self.registry.register_canary(
                        model=new_model,
                        metrics=new_metrics,
                        history=history,
                        model_params=self._model_params(),
                    )
                    run_status = "canary_registered"
                else:
                    registration = self.registry.register_model(
                        model=new_model,
                        metrics=new_metrics,
                        history=history,
                        model_params=self._model_params(),
                    )
                    run_status = "promoted"
                    self.baseline_metrics = new_metrics
                self._log_run(new_metrics, status=run_status, model=new_model, data=data)
                self.metrics_exporter.set_model_metrics(
                    rmse=float(new_metrics["rmse"]),
                    r2=float(new_metrics["r2"]),
                )
                logger.info(
                    "Registered model version %s (%s)",
                    registration["version"],
                    run_status,
                )
                return True

            self._log_run(new_metrics, status="rejected", reason="no_improvement")
            logger.info("Model rejected: no significant improvement over baseline")
            return False

        except Exception as exc:
            logger.exception("Retraining failed: %s", exc)
            self._send_alert(f"Retraining failed: {exc}")
            return False

    def _train_and_evaluate(self) -> Dict[str, Any]:
        pipeline = TrainingPipeline(self.config)
        pipeline.prepare_data()
        pipeline.build_model()
        pipeline.train_model()
        _, predicted_prices = pipeline.evaluate_model()

        history_dict = pipeline.trainer.get_training_history() if pipeline.trainer else {}

        return {
            "model": pipeline.model,
            "history": history_dict,
            "metrics": pipeline.metrics,
            "predicted_prices": predicted_prices,
            "data": pipeline.data,
        }

    def _should_promote(self, new_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> bool:
        # Roadmap gating: lower errors + higher R2.
        return (
            float(new_metrics["rmse"]) < float(baseline_metrics["rmse"])
            and float(new_metrics["r2"]) > float(baseline_metrics["r2"])
            and float(new_metrics["mape"]) <= float(baseline_metrics["mape"])
        )

    def _model_params(self) -> Dict[str, Any]:
        model_cfg = getattr(self.config, "model", None)
        if model_cfg is None:
            return {}
        return {
            "sequence_length": getattr(model_cfg, "SEQUENCE_LENGTH", None),
            "lstm_units_1": getattr(model_cfg, "LSTM_UNITS_1", None),
            "lstm_units_2": getattr(model_cfg, "LSTM_UNITS_2", None),
            "dense_units": getattr(model_cfg, "DENSE_UNITS", None),
            "dropout_rate": getattr(model_cfg, "DROPOUT_RATE", None),
            "learning_rate": getattr(model_cfg, "LEARNING_RATE", None),
            "epochs": getattr(model_cfg, "EPOCHS", None),
            "batch_size": getattr(model_cfg, "BATCH_SIZE", None),
        }

    def _log_run(
        self,
        metrics: Dict[str, Any],
        status: str,
        reason: Optional[str] = None,
        model=None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        tagged_metrics: Dict[str, Any] = dict(metrics)
        run_params = self._model_params()
        run_params["run_status"] = status
        if reason:
            run_params["run_reason"] = reason

        self.tracker.log_training(
            params=run_params,
            metrics=tagged_metrics,
            model=model,
            X_test=data.get("X_test") if data else None,
            y_test=data.get("y_test") if data else None,
        )

        model_version = None
        try:
            production = self.registry.get_current_production_model()
            model_version = production.get("version") if production else None
        except Exception:
            model_version = None

        try:
            self.audit_store.log_training_run(
                status=status,
                trigger="scheduler_or_manual",
                reason=reason,
                metrics=tagged_metrics,
                model_version=model_version,
            )
        except Exception as exc:
            logger.warning("Failed to persist retraining audit record: %s", exc)

    def _send_alert(self, message: str) -> None:
        self.alerting.send(
            title="Retraining failure",
            message=message,
            level="critical",
            alert_key="retraining_failure",
        )

    def schedule_weekly(self) -> None:
        schedule.every().sunday.at("02:00").do(self.trigger_retraining)
        logger.info("Weekly retraining scheduled for Sunday 02:00")

    def start_scheduler(self) -> None:
        self.schedule_weekly()
        while True:
            schedule.run_pending()
            time.sleep(60)
