from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.settings import Config
from src.mlops.background_tasks import SchedulerBackgroundService
from src.mlops.metrics_exporter import MetricsExporter
from src.mlops.retraining_pipeline import AutoRetrainingPipeline
from src.mlops.scheduler import MLOpsScheduler


@dataclass
class MLOpsIntegration:
    """Runtime holder for all Phase 4 MLOps components."""

    config: Config
    metrics_exporter: MetricsExporter
    pipeline: AutoRetrainingPipeline
    scheduler: MLOpsScheduler
    background_service: SchedulerBackgroundService

    def start(self) -> None:
        self.background_service.start()

    def stop(self, timeout_seconds: float = 10.0) -> None:
        self.background_service.stop(timeout_seconds=timeout_seconds)

    def health(self) -> Dict[str, Any]:
        production_model = self.pipeline.registry.get_current_production_model()
        return {
            "status": "healthy",
            "scheduler": self.background_service.status(),
            "drift_last_report": self.pipeline.drift_detector.get_drift_report(),
            "baseline_metrics": self.pipeline.baseline_metrics,
            "production_model": {
                "name": production_model.get("name"),
                "version": production_model.get("version"),
                "timestamp": production_model.get("timestamp"),
            }
            if production_model
            else None,
        }

    def list_registry(self, name: Optional[str] = None):
        return self.pipeline.registry.list_all(name=name)


def initialize_mlops(
    config: Optional[Config] = None,
    scheduler_poll_seconds: int = 60,
    start_prometheus_http_server: bool = False,
    prometheus_port: int = 8001,
) -> MLOpsIntegration:
    """Create MLOps components and run scheduler in background thread."""
    resolved_config = config or Config()

    metrics_exporter = MetricsExporter()
    if start_prometheus_http_server:
        metrics_exporter.start_server(port=prometheus_port)

    pipeline = AutoRetrainingPipeline(resolved_config)
    pipeline.metrics_exporter = metrics_exporter

    scheduler = MLOpsScheduler(
        retraining_pipeline=pipeline,
        drift_detector=pipeline.drift_detector,
    )

    background_service = SchedulerBackgroundService(
        scheduler=scheduler,
        poll_seconds=scheduler_poll_seconds,
    )

    integration = MLOpsIntegration(
        config=resolved_config,
        metrics_exporter=metrics_exporter,
        pipeline=pipeline,
        scheduler=scheduler,
        background_service=background_service,
    )

    integration.start()
    return integration
