from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate model quality and prediction sanity before promotion."""

    def __init__(self, thresholds=None):
        # Default price range tuned for IRR-scale gold values.
        self.thresholds = thresholds or {
            "rmse_max": 120000,
            "mae_max": 90000,
            "r2_min": 0.70,
            "mape_max": 25.0,
            "prediction_min": 10_000_000,
            "prediction_max": 1_000_000_000,
        }

    def validate_metrics(self, metrics: Dict) -> Tuple[bool, Dict[str, bool]]:
        required = ("rmse", "mae", "r2", "mape")
        missing = [name for name in required if name not in metrics]
        if missing:
            return False, {f"{name}_present": False for name in missing}

        rmse = float(metrics["rmse"])
        mae = float(metrics["mae"])
        r2 = float(metrics["r2"])
        mape = float(metrics["mape"])

        checks = {
            "rmse_finite": np.isfinite(rmse),
            "mae_finite": np.isfinite(mae),
            "r2_finite": np.isfinite(r2),
            "mape_finite": np.isfinite(mape),
            "rmse_valid": rmse <= self.thresholds["rmse_max"],
            "mae_valid": mae <= self.thresholds["mae_max"],
            "r2_valid": r2 >= self.thresholds["r2_min"],
            "mape_valid": mape <= self.thresholds["mape_max"],
        }
        passed = all(checks.values())
        if not passed:
            logger.warning("Metric validation failed: %s", checks)
        return passed, checks

    def validate_model_weights(self, model) -> Tuple[bool, str]:
        try:
            for layer in model.layers:
                for weight in layer.get_weights():
                    if np.any(np.isnan(weight)) or np.any(np.isinf(weight)):
                        return False, f"Invalid weights detected in layer '{layer.name}'"
            return True, "weights_valid"
        except Exception as exc:
            return False, str(exc)

    def validate_predictions(
        self,
        predictions: np.ndarray,
        reference_prices: np.ndarray | None = None,
    ) -> Tuple[bool, str]:
        preds = np.asarray(predictions).astype(float).flatten()
        if preds.size == 0:
            return False, "No predictions generated"
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            return False, "Predictions contain NaN/Inf"

        if reference_prices is not None:
            ref = np.asarray(reference_prices).astype(float).flatten()
            ref = ref[np.isfinite(ref)]
            if ref.size:
                ref_min = float(np.min(ref))
                ref_max = float(np.max(ref))
                # Keep a generous 50% band around recent real prices.
                dynamic_min = max(self.thresholds["prediction_min"], ref_min * 0.5)
                dynamic_max = min(self.thresholds["prediction_max"], ref_max * 1.5)
            else:
                dynamic_min = self.thresholds["prediction_min"]
                dynamic_max = self.thresholds["prediction_max"]
        else:
            dynamic_min = self.thresholds["prediction_min"]
            dynamic_max = self.thresholds["prediction_max"]

        pred_min = float(np.min(preds))
        pred_max = float(np.max(preds))
        if pred_min < dynamic_min or pred_max > dynamic_max:
            return False, f"Predictions out of range [{pred_min}, {pred_max}] expected [{dynamic_min}, {dynamic_max}]"

        return True, "predictions_valid"

    def full_validation(
        self,
        model,
        metrics: Dict,
        predictions: np.ndarray,
        reference_prices: np.ndarray | None = None,
    ) -> Tuple[bool, Dict[str, str]]:
        metrics_ok, metric_report = self.validate_metrics(metrics)
        weights_ok, weight_report = self.validate_model_weights(model)
        preds_ok, pred_report = self.validate_predictions(
            predictions=predictions,
            reference_prices=reference_prices,
        )

        report = {
            "metrics": str(metric_report),
            "weights": weight_report,
            "predictions": pred_report,
        }

        return metrics_ok and weights_ok and preds_ok, report
