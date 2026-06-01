from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from src.api.predictor import GoldPricePredictor
from src.mlops.ab_testing import ABTestRouter
from src.mlops.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class PredictionRouter:
    """Route inference to production or canary predictors."""

    def __init__(
        self,
        *,
        default_predictor: GoldPricePredictor,
        registry: ModelRegistry,
        scaler_X_path: str,
        scaler_y_path: str,
        ab_router: Optional[ABTestRouter] = None,
    ):
        self.default_predictor = default_predictor
        self.registry = registry
        self.scaler_X_path = scaler_X_path
        self.scaler_y_path = scaler_y_path
        self.ab_router = ab_router or ABTestRouter(registry=registry)
        self._canary_cache: Dict[str, GoldPricePredictor] = {}

    def resolve(
        self,
        routing_key: Optional[str] = None,
    ) -> Tuple[GoldPricePredictor, Dict[str, Any]]:
        variant = self.ab_router.select_variant(routing_key)
        entry = self.ab_router.resolve_model_entry(variant)

        if variant == "production" or entry is None:
            return self.default_predictor, {
                "variant": "production",
                "model_version": None,
                "model_path": self.default_predictor.model_path,
            }

        path = entry.get("path")
        if not path:
            return self.default_predictor, {
                "variant": "production",
                "model_version": None,
                "model_path": self.default_predictor.model_path,
            }

        predictor = self._canary_cache.get(path)
        if predictor is None:
            predictor = GoldPricePredictor(
                model_path=path,
                scaler_X_path=self.scaler_X_path,
                scaler_y_path=self.scaler_y_path,
            )
            self._canary_cache[path] = predictor
            logger.info("Loaded canary predictor from %s", path)

        return predictor, {
            "variant": "canary",
            "model_version": entry.get("version"),
            "model_path": path,
        }

    def ab_status(self) -> Dict[str, Any]:
        return self.ab_router.status()
