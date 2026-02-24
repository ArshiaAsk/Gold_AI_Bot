import asyncio

from src.api.main import metrics, mlops_artifacts, mlops_health, mlops_model_registry
from src.mlops.metrics_exporter import MetricsExporter


class DummyIntegration:
    def __init__(self):
        self.metrics_exporter = MetricsExporter()

    def health(self):
        return {
            "status": "healthy",
            "scheduler": {"running": True},
            "drift_last_report": None,
            "baseline_metrics": {"rmse": 1.0, "r2": 0.9, "mape": 1.0},
            "production_model": {"name": "gold_lstm", "version": 1, "timestamp": "20260223T000000Z"},
        }

    def list_registry(self, name=None):
        models = [{"name": "gold_lstm", "version": 1, "status": "production"}]
        if name is None:
            return models
        return [m for m in models if m["name"] == name]


def test_mlops_endpoints_without_lifespan():
    original_integration = mlops_artifacts.get("integration")
    mlops_artifacts["integration"] = DummyIntegration()

    try:
        health_payload = asyncio.run(mlops_health())
        registry_payload = asyncio.run(mlops_model_registry())
        metrics_response = asyncio.run(metrics())

        assert health_payload["status"] == "healthy"
        assert registry_payload["models"][0]["name"] == "gold_lstm"
        assert metrics_response.status_code == 200
        assert b"# HELP" in metrics_response.body
    finally:
        mlops_artifacts["integration"] = original_integration
