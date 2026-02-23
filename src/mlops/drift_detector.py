from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for lightweight drift detection."""

    p_value_threshold: float = 0.05
    mean_shift_threshold: float = 0.2


class DriftDetector:
    """Detect drift using Evidently when available, otherwise a lightweight fallback."""

    def __init__(self, reference_data: Optional[Any] = None, config: Optional[DriftConfig] = None):
        self.reference_data = self._to_dataframe(reference_data) if reference_data is not None else None
        self.config = config or DriftConfig()
        self.last_report: Optional[Dict[str, Any]] = None

    def set_reference_data(self, reference_data: Any) -> None:
        self.reference_data = self._to_dataframe(reference_data)

    def check_drift(self, current_data: Any) -> Dict[str, Any]:
        """Run drift check and return a normalized report."""
        current_df = self._to_dataframe(current_data)

        if self.reference_data is None:
            # First run: establish baseline.
            self.reference_data = current_df.copy()
            self.last_report = {
                "drift_detected": False,
                "drift_score": 0.0,
                "method": "bootstrap_reference",
                "details": "Reference data initialized from current batch.",
            }
            return self.last_report

        try:
            report = self._check_with_evidently(self.reference_data, current_df)
        except Exception as exc:
            logger.warning("Evidently drift check unavailable, using fallback: %s", exc)
            report = self._check_with_statistics(self.reference_data, current_df)

        self.last_report = report
        return report

    def get_drift_report(self) -> Optional[Dict[str, Any]]:
        return self.last_report

    def _check_with_evidently(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        result = report.as_dict()

        metrics = result.get("metrics", [])
        drift_score = 0.0
        drift_detected = False

        if metrics:
            metric_result = metrics[0].get("result", {})
            drift_score = float(metric_result.get("share_of_drifted_columns", 0.0))
            drift_detected = bool(metric_result.get("dataset_drift", False))

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "method": "evidently",
            "raw_report": result,
        }

    def _check_with_statistics(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        shared_cols = [c for c in reference_df.columns if c in current_df.columns]
        if not shared_cols:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "method": "fallback",
                "details": "No shared columns to compare.",
            }

        drifted = 0
        col_results: Dict[str, Dict[str, float]] = {}

        for col in shared_cols:
            ref = pd.to_numeric(reference_df[col], errors="coerce").dropna().values
            cur = pd.to_numeric(current_df[col], errors="coerce").dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue

            p_value = float(ks_2samp(ref, cur).pvalue)
            ref_std = float(np.std(ref)) or 1.0
            mean_shift = abs(float(np.mean(cur) - np.mean(ref))) / ref_std
            is_drift = (p_value < self.config.p_value_threshold) and (
                mean_shift > self.config.mean_shift_threshold
            )
            drifted += int(is_drift)
            col_results[col] = {
                "p_value": p_value,
                "mean_shift_sigma": mean_shift,
                "drift": float(is_drift),
            }

        total = max(len(col_results), 1)
        drift_score = drifted / total

        return {
            "drift_detected": drifted > 0,
            "drift_score": drift_score,
            "method": "fallback",
            "column_results": col_results,
        }

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()

        arr = np.asarray(data)
        if arr.ndim == 3:
            # Flatten time dimension for comparison.
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        cols = [f"feature_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
