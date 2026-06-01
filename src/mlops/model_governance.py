from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.mlops.model_registry import ModelRegistry


class ModelCardGenerator:
    """Generate governance model cards from registry metadata."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        cards_dir: str = "models/model_cards",
    ):
        self.registry = registry or ModelRegistry()
        self.cards_dir = Path(cards_dir)
        self.cards_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        version: Optional[int] = None,
        name: str = "gold_lstm",
        owner: str = "Data Science Team",
        business_impact: str = "Trading signal accuracy for gold price forecasts",
    ) -> Dict[str, Any]:
        entry = self._resolve_entry(version=version, name=name)
        if entry is None:
            raise ValueError(f"No model found for name={name!r} version={version!r}")

        metrics = entry.get("metrics") or {}
        params = entry.get("model_params") or {}
        card = {
            "model_name": name,
            "version": entry.get("version"),
            "status": entry.get("status"),
            "architecture": {
                "type": "LSTM",
                "lstm_units_1": params.get("lstm_units_1"),
                "lstm_units_2": params.get("lstm_units_2"),
                "dense_units": params.get("dense_units"),
                "dropout_rate": params.get("dropout_rate"),
                "sequence_length": params.get("sequence_length", 30),
                "n_features": 15,
            },
            "training_data": {
                "source": "data/raw/final_gold_dataset.csv",
                "registered_at": entry.get("timestamp"),
            },
            "performance": {
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "r2": metrics.get("r2"),
                "mape": metrics.get("mape"),
            },
            "limitations": [
                "Not tuned for extreme market shocks or regulatory regime changes.",
                "Assumes historical feature correlations remain stable.",
                "Requires periodic retraining when drift is detected.",
            ],
            "governance": {
                "owner": owner,
                "approval_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "business_impact": business_impact,
                "ethical_considerations": "No sensitive personal data used in training.",
            },
            "artifact_path": entry.get("path"),
        }

        output_path = self.cards_dir / f"{name}_v{entry.get('version')}_model_card.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(card, handle, indent=2, ensure_ascii=True)
        card["card_path"] = str(output_path)
        return card

    def list_cards(self) -> List[str]:
        return sorted(str(path) for path in self.cards_dir.glob("*_model_card.json"))

    def _resolve_entry(self, *, version: Optional[int], name: str) -> Optional[Dict[str, Any]]:
        if version is not None:
            for entry in self.registry.list_all(name=name):
                if int(entry.get("version", -1)) == int(version):
                    return entry
            return None

        production = self.registry.get_current_production_model(name=name)
        if production:
            return production
        return self.registry.get_best_model(name=name)
