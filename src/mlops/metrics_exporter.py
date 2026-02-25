from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server


@dataclass
class MetricsExporter:
    """Small helper for writing Prometheus metrics consistently."""

    started: bool = False
    registry: CollectorRegistry | None = None

    def __post_init__(self) -> None:
        self.registry = self.registry or CollectorRegistry(auto_describe=True)
        self.prediction_latency = Histogram(
            "prediction_latency_seconds",
            "Prediction latency in seconds",
            registry=self.registry,
        )
        self.signal_counter = Counter(
            "trading_signals_total",
            "Trading signals count",
            ["action"],
            registry=self.registry,
        )
        self.confidence_gauge = Gauge(
            "signal_confidence",
            "Latest signal confidence (0-1)",
            registry=self.registry,
        )
        self.model_rmse_gauge = Gauge(
            "model_accuracy_rmse",
            "Current production model RMSE",
            registry=self.registry,
        )
        self.model_r2_gauge = Gauge(
            "model_accuracy_r2",
            "Current production model R2",
            registry=self.registry,
        )
        self.model_drift_counter = Counter(
            "model_drift_detected_total",
            "Data drift detection count",
            registry=self.registry,
        )
        self.api_errors_counter = Counter(
            "api_errors_total",
            "Total API errors",
            registry=self.registry,
        )
        self.api_uptime_gauge = Gauge(
            "api_uptime_ratio",
            "Service uptime ratio from 0 to 1",
            registry=self.registry,
        )
        self.prediction_sla_violations = Counter(
            "prediction_sla_violations_total",
            "Prediction requests violating SLA latency threshold",
            registry=self.registry,
        )
        self.prediction_sla_seconds = float(os.getenv("PREDICTION_SLA_SECONDS", "1.0"))

    def start_server(self, port: int = 8001) -> None:
        if self.started:
            return
        start_http_server(port, registry=self.registry)
        self.started = True

    def record_prediction(self, latency_seconds: float, confidence: Optional[float] = None) -> None:
        latency = float(latency_seconds)
        self.prediction_latency.observe(latency)
        if latency > self.prediction_sla_seconds:
            self.prediction_sla_violations.inc()
        if confidence is not None:
            self.confidence_gauge.set(float(confidence))

    def increment_signal(self, action: str) -> None:
        self.signal_counter.labels(action=action.upper()).inc()

    def set_model_metrics(self, rmse: float, r2: float) -> None:
        self.model_rmse_gauge.set(float(rmse))
        self.model_r2_gauge.set(float(r2))

    def record_drift(self) -> None:
        self.model_drift_counter.inc()

    def record_api_error(self) -> None:
        self.api_errors_counter.inc()

    def set_api_uptime_ratio(self, value: float) -> None:
        self.api_uptime_gauge.set(max(0.0, min(1.0, float(value))))
