from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import schedule

logger = logging.getLogger(__name__)


@dataclass
class MLOpsScheduler:
    """Schedule periodic MLOps tasks."""

    retraining_pipeline: object
    drift_detector: object | None = None
    daily_drift_data_supplier: Optional[Callable[[], Any]] = None

    def __post_init__(self) -> None:
        self._scheduler = schedule.Scheduler()

    def configure(self) -> None:
        self._scheduler.every().sunday.at("02:00").do(self.retraining_pipeline.trigger_retraining)

        if self.drift_detector is not None:
            self._scheduler.every().day.at("06:00").do(self._run_daily_drift_check)

        logger.info("MLOps scheduler configured")

    def run_forever(self, poll_seconds: int = 60) -> None:
        self.configure()
        while True:
            self._scheduler.run_pending()
            time.sleep(poll_seconds)

    def _run_daily_drift_check(self):
        if self.drift_detector is None:
            return None

        try:
            if self.daily_drift_data_supplier is not None:
                current_data = self.daily_drift_data_supplier()
                report = self.drift_detector.check_drift(current_data)
            else:
                report = self.drift_detector.get_drift_report()

            logger.info("Daily drift check completed. Report: %s", report)
            return report
        except Exception as exc:
            logger.exception("Daily drift check failed: %s", exc)
            return None
