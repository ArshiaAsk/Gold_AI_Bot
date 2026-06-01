from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

import keras

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Filesystem-backed model registry with version metadata and rollback."""

    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.registry_path / "versions"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "metadata.json"
        self.metadata: List[Dict[str, Any]] = self._load_metadata()

    def register_model(
        self,
        model,
        metrics: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "gold_lstm",
    ) -> Dict[str, Any]:
        version = self._next_version(name)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_path = self.models_dir / f"{name}_v{version}_{timestamp}.keras"
        model.save(str(model_path))

        # Only one model should be active production at once.
        for entry in self.metadata:
            if entry.get("name") == name and entry.get("status") == "production":
                entry["status"] = "archived"
                entry["promoted"] = False

        serializable_metrics = self._to_serializable(metrics)
        serializable_history = self._to_serializable(history)
        serializable_params = self._to_serializable(model_params)

        entry: Dict[str, Any] = {
            "version": version,
            "name": name,
            "path": str(model_path),
            "timestamp": timestamp,
            "metrics": serializable_metrics,
            "model_params": serializable_params,
            "training_history": serializable_history,
            "promoted": True,
            "status": "production",
        }

        self.metadata.append(entry)
        self._save_metadata()
        logger.info("Registered model %s version %s", name, version)
        return entry

    def register_canary(
        self,
        model,
        metrics: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        name: str = "gold_lstm",
    ) -> Dict[str, Any]:
        """Register a candidate model for A/B canary traffic."""
        self.clear_canary(name=name)
        entry = self._save_versioned_model(
            model=model,
            metrics=metrics,
            history=history,
            model_params=model_params,
            name=name,
            status="canary",
            promoted=False,
        )
        logger.info("Registered canary model %s version %s", name, entry["version"])
        return entry

    def get_canary_model(self, name: str = "gold_lstm") -> Optional[Dict[str, Any]]:
        for entry in reversed(self.metadata):
            if entry.get("name") == name and entry.get("status") == "canary":
                return entry
        return None

    def promote_canary(self, name: str = "gold_lstm") -> Dict[str, Any]:
        canary = self.get_canary_model(name=name)
        if canary is None:
            raise ValueError(f"No canary model registered for '{name}'")

        for entry in self.metadata:
            if entry.get("name") != name:
                continue
            if entry.get("version") == canary.get("version"):
                entry["status"] = "production"
                entry["promoted"] = True
            elif entry.get("status") == "production":
                entry["status"] = "archived"
                entry["promoted"] = False

        self._save_metadata()
        logger.info("Promoted canary %s version %s to production", name, canary.get("version"))
        return canary

    def clear_canary(self, name: str = "gold_lstm") -> None:
        changed = False
        for entry in self.metadata:
            if entry.get("name") == name and entry.get("status") == "canary":
                entry["status"] = "archived"
                entry["promoted"] = False
                changed = True
        if changed:
            self._save_metadata()
            logger.info("Cleared canary model for %s", name)

    def get_best_model(self, name: str = "gold_lstm") -> Optional[Dict[str, Any]]:
        candidates = [m for m in self.metadata if m.get("name") == name]
        if not candidates:
            return None

        def r2_score(entry: Dict[str, Any]) -> float:
            try:
                value = float(entry.get("metrics", {}).get("r2", float("-inf")))
                return value if value == value else float("-inf")
            except Exception:
                return float("-inf")

        return max(candidates, key=r2_score)

    def get_current_production_model(self, name: str = "gold_lstm") -> Optional[Dict[str, Any]]:
        for entry in reversed(self.metadata):
            if entry.get("name") == name and entry.get("status") == "production":
                return entry
        return None

    def get_model_path(self, version: Optional[int] = None, name: str = "gold_lstm") -> Optional[str]:
        if version is None:
            entry = self.get_current_production_model(name)
        else:
            entry = next(
                (m for m in self.metadata if m.get("name") == name and m.get("version") == version),
                None,
            )
        return entry.get("path") if entry else None

    def load_model(self, version: Optional[int] = None, name: str = "gold_lstm"):
        path = self.get_model_path(version=version, name=name)
        return keras.models.load_model(path) if path else None

    def rollback(self, target_version: int, name: str = "gold_lstm") -> Dict[str, Any]:
        target = None
        for entry in self.metadata:
            if entry.get("name") != name:
                continue
            if entry.get("version") == target_version:
                target = entry
                entry["status"] = "production"
                entry["promoted"] = True
            else:
                entry["status"] = "archived"
                entry["promoted"] = False

        if target is None:
            raise ValueError(f"Version {target_version} not found for model '{name}'")

        self._save_metadata()
        logger.info("Rolled back model %s to version %s", name, target_version)
        return target

    def list_all(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        if name is None:
            return list(self.metadata)
        return [dict(m) for m in self.metadata if m.get("name") == name]

    def get_metrics_history(self, name: str = "gold_lstm") -> Dict[int, Dict[str, Any]]:
        return {
            int(entry["version"]): entry.get("metrics", {})
            for entry in self.metadata
            if entry.get("name") == name
        }

    def _next_version(self, name: str) -> int:
        versions = [int(m["version"]) for m in self.metadata if m.get("name") == name]
        return max(versions, default=0) + 1

    def _save_versioned_model(
        self,
        *,
        model,
        metrics: Dict[str, Any],
        history: Optional[Dict[str, Any]],
        model_params: Optional[Dict[str, Any]],
        name: str,
        status: str,
        promoted: bool,
    ) -> Dict[str, Any]:
        version = self._next_version(name)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        model_path = self.models_dir / f"{name}_v{version}_{timestamp}.keras"
        model.save(str(model_path))

        entry: Dict[str, Any] = {
            "version": version,
            "name": name,
            "path": str(model_path),
            "timestamp": timestamp,
            "metrics": self._to_serializable(metrics),
            "model_params": self._to_serializable(model_params),
            "training_history": self._to_serializable(history),
            "promoted": promoted,
            "status": status,
        }
        self.metadata.append(entry)
        self._save_metadata()
        return entry

    def _load_metadata(self) -> List[Dict[str, Any]]:
        if not self.metadata_file.exists():
            return []
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error("Failed to read registry metadata (%s). Starting empty.", exc)
            return []

    def _save_metadata(self) -> None:
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=str(self.metadata_file.parent), delete=False, encoding="utf-8") as tmp:
            json.dump(self.metadata, tmp, indent=2, ensure_ascii=True)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.metadata_file)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: ModelRegistry._to_serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ModelRegistry._to_serializable(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value
