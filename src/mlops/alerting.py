from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Alerting:
    """Alert facade with basic dedupe/cooldown to reduce noisy notifications."""

    service_name: str = "gold-mlops"
    cooldown_seconds: int = 300
    _last_sent_at: dict[str, float] = field(default_factory=dict)

    def should_send(self, alert_key: str) -> bool:
        now = time.time()
        last = self._last_sent_at.get(alert_key, 0.0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_sent_at[alert_key] = now
        return True

    def send(self, title: str, message: str, level: str = "error", alert_key: str | None = None) -> None:
        key = alert_key or title.lower().strip().replace(" ", "_")
        if not self.should_send(key):
            logger.info("[%s] Alert suppressed by cooldown: %s", self.service_name, key)
            return

        payload = f"[{self.service_name}] {title}: {message}"
        if level == "critical":
            logger.critical(payload)
        elif level == "warning":
            logger.warning(payload)
        else:
            logger.error(payload)
