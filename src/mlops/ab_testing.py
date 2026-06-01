from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.mlops.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """Runtime configuration for canary A/B routing."""

    enabled: bool = False
    canary_traffic_percent: float = 0.0

    @classmethod
    def from_env(cls) -> "ABTestConfig":
        enabled = os.getenv("AB_TESTING_ENABLED", "false").lower() in {"1", "true", "yes"}
        try:
            percent = float(os.getenv("AB_CANARY_TRAFFIC_PERCENT", "10"))
        except ValueError:
            percent = 10.0
        percent = max(0.0, min(100.0, percent))
        return cls(enabled=enabled, canary_traffic_percent=percent)


@dataclass
class ABTestRouter:
    """Deterministic traffic splitter between production and canary models."""

    registry: ModelRegistry
    config: ABTestConfig = field(default_factory=ABTestConfig.from_env)

    def select_variant(self, routing_key: Optional[str] = None) -> str:
        """
        Return 'production' or 'canary'.

        Uses stable hashing so the same routing_key always hits the same variant.
        """
        if not self.config.enabled or self.config.canary_traffic_percent <= 0:
            return "production"

        canary = self.registry.get_canary_model()
        if canary is None:
            return "production"

        key = routing_key or "default"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        if bucket < int(self.config.canary_traffic_percent):
            return "canary"
        return "production"

    def resolve_model_entry(self, variant: str) -> Optional[Dict[str, Any]]:
        if variant == "canary":
            return self.registry.get_canary_model()
        return self.registry.get_current_production_model()

    def status(self) -> Dict[str, Any]:
        production = self.registry.get_current_production_model()
        canary = self.registry.get_canary_model()
        return {
            "enabled": self.config.enabled,
            "canary_traffic_percent": self.config.canary_traffic_percent,
            "production": _summarize_entry(production),
            "canary": _summarize_entry(canary),
        }

    def promote_canary(self) -> Dict[str, Any]:
        return self.registry.promote_canary()

    def disable_canary(self) -> None:
        self.registry.clear_canary()


def _summarize_entry(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if entry is None:
        return None
    return {
        "version": entry.get("version"),
        "status": entry.get("status"),
        "timestamp": entry.get("timestamp"),
        "metrics": entry.get("metrics"),
    }
