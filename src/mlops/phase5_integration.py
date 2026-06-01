from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.mlops.ab_testing import ABTestRouter
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.model_governance import ModelCardGenerator
from src.mlops.model_registry import ModelRegistry
from src.mlops.onnx_exporter import ensure_onnx_artifact


@dataclass
class Phase5Integration:
    """Phase 5: experiment tracking, A/B testing, governance, ONNX optimization."""

    registry: ModelRegistry
    experiment_tracker: ExperimentTracker
    ab_router: ABTestRouter
    model_cards: ModelCardGenerator

    @classmethod
    def create(cls, registry: Optional[ModelRegistry] = None) -> "Phase5Integration":
        resolved_registry = registry or ModelRegistry()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "gold_price_retraining")
        return cls(
            registry=resolved_registry,
            experiment_tracker=ExperimentTracker(experiment_name, tracking_uri=tracking_uri),
            ab_router=ABTestRouter(registry=resolved_registry),
            model_cards=ModelCardGenerator(registry=resolved_registry),
        )

    def status(self) -> Dict[str, Any]:
        return {
            "experiment_tracking": {
                "enabled": self.experiment_tracker.enabled,
                "experiment_name": self.experiment_tracker.experiment_name,
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
            },
            "ab_testing": self.ab_router.status(),
        }

    def list_experiments(self, max_results: int = 10) -> List[Dict[str, Any]]:
        runs = self.experiment_tracker.compare_experiments(max_results=max_results)
        if not runs:
            return []

        if hasattr(runs, "to_dict"):
            records = runs.to_dict(orient="records")
            return records[:max_results]
        return []

    def generate_model_card(
        self,
        *,
        version: Optional[int] = None,
        name: str = "gold_lstm",
    ) -> Dict[str, Any]:
        return self.model_cards.generate(version=version, name=name)

    def list_model_cards(self) -> List[str]:
        return self.model_cards.list_cards()

    def export_onnx(self, keras_model_path: str) -> Optional[str]:
        if os.getenv("ONNX_EXPORT_ENABLED", "true").lower() not in {"1", "true", "yes"}:
            return None
        return ensure_onnx_artifact(keras_model_path)
