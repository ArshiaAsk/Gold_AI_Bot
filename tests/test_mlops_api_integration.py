from config.settings import Config
from src.mlops import api_integration


class DummyDriftDetector:
    def get_drift_report(self):
        return {"drift_detected": False}


class DummyRegistry:
    def __init__(self):
        self.models = [{"name": "gold_lstm", "version": 1, "status": "production"}]

    def get_current_production_model(self):
        return {"name": "gold_lstm", "version": 1, "timestamp": "20260223T000000Z"}

    def get_canary_model(self):
        return None

    def list_all(self, name=None):
        if name is None:
            return list(self.models)
        return [m for m in self.models if m.get("name") == name]


class DummyPipeline:
    def __init__(self, config):
        self.config = config
        self.drift_detector = DummyDriftDetector()
        self.registry = DummyRegistry()
        self.baseline_metrics = {"rmse": 1.0, "r2": 0.9, "mape": 1.0}
        self.metrics_exporter = None

    def trigger_retraining(self):
        return True


class DummyScheduler:
    def __init__(self, retraining_pipeline, drift_detector=None, daily_drift_data_supplier=None):
        self.retraining_pipeline = retraining_pipeline
        self.drift_detector = drift_detector
        self.daily_drift_data_supplier = daily_drift_data_supplier
        self.configure_calls = 0

    def configure(self):
        self.configure_calls += 1

    def run_pending_once(self):
        return None


def test_initialize_mlops_and_health(monkeypatch):
    monkeypatch.setattr(api_integration, "AutoRetrainingPipeline", DummyPipeline)
    monkeypatch.setattr(api_integration, "MLOpsScheduler", DummyScheduler)

    integration = api_integration.initialize_mlops(config=Config(), scheduler_poll_seconds=0)
    try:
        health = integration.health()
        models = integration.list_registry()

        assert health["status"] == "healthy"
        assert health["scheduler"]["running"] is True
        assert health["production_model"]["version"] == 1
        assert models and models[0]["name"] == "gold_lstm"
    finally:
        integration.stop(timeout_seconds=1)
