import hashlib
from unittest.mock import MagicMock

import pytest

from src.mlops.ab_testing import ABTestConfig, ABTestRouter
from src.mlops.model_governance import ModelCardGenerator
from src.mlops.model_registry import ModelRegistry
from src.mlops.prediction_router import PredictionRouter


@pytest.fixture
def registry_with_canary(tmp_path):
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))
    registry.metadata = [
        {
            "version": 1,
            "name": "gold_lstm",
            "path": "/tmp/v1.keras",
            "timestamp": "20260101T000000Z",
            "metrics": {"rmse": 100.0, "mae": 80.0, "r2": 0.9, "mape": 5.0},
            "status": "production",
            "promoted": True,
        },
        {
            "version": 2,
            "name": "gold_lstm",
            "path": "/tmp/v2.keras",
            "timestamp": "20260102T000000Z",
            "metrics": {"rmse": 90.0, "mae": 70.0, "r2": 0.92, "mape": 4.5},
            "status": "canary",
            "promoted": False,
        },
    ]
    registry._save_metadata()
    return registry


def test_ab_router_stable_assignment(registry_with_canary):
    config = ABTestConfig(enabled=True, canary_traffic_percent=50.0)
    router = ABTestRouter(registry=registry_with_canary, config=config)

    first = router.select_variant("client-a")
    second = router.select_variant("client-a")
    assert first == second


def test_ab_router_respects_traffic_percent(registry_with_canary):
    config = ABTestConfig(enabled=True, canary_traffic_percent=0.0)
    router = ABTestRouter(registry=registry_with_canary, config=config)
    assert router.select_variant("any-client") == "production"


def test_promote_canary(registry_with_canary):
    promoted = registry_with_canary.promote_canary()
    assert promoted["version"] == 2
    production = registry_with_canary.get_current_production_model()
    assert production["version"] == 2
    assert registry_with_canary.get_canary_model() is None


def test_model_card_generation(registry_with_canary, tmp_path):
    generator = ModelCardGenerator(
        registry=registry_with_canary,
        cards_dir=str(tmp_path / "cards"),
    )
    card = generator.generate(version=1, name="gold_lstm")
    assert card["version"] == 1
    assert card["performance"]["rmse"] == 100.0
    assert (tmp_path / "cards").exists()


def test_prediction_router_falls_back_without_canary(tmp_path):
    registry = ModelRegistry(registry_path=str(tmp_path / "registry"))
    default_predictor = MagicMock()
    default_predictor.model_path = "models/gold_lstm_v2.keras"

    router = PredictionRouter(
        default_predictor=default_predictor,
        registry=registry,
        scaler_X_path="models/scaler_X.pkl",
        scaler_y_path="models/scaler_y.pkl",
    )
    predictor, meta = router.resolve("user-1")
    assert predictor is default_predictor
    assert meta["variant"] == "production"


def test_hash_bucket_distribution():
    buckets = []
    for idx in range(1000):
        digest = hashlib.sha256(f"user-{idx}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        buckets.append(bucket)
    canary_hits = sum(1 for b in buckets if b < 10)
    assert 50 < canary_hits < 150
