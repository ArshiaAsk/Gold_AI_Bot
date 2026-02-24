from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SchedulerBackgroundService:
    """Run MLOps scheduler in a dedicated background thread."""

    scheduler: object
    poll_seconds: int = 60
    name: str = "mlops-scheduler"
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self.scheduler.configure()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()
        logger.info("Started MLOps scheduler background thread '%s'", self.name)

    def stop(self, timeout_seconds: float = 10.0) -> None:
        if not self._thread:
            return

        self._stop_event.set()
        self._thread.join(timeout=timeout_seconds)
        logger.info("Stopped MLOps scheduler background thread '%s'", self.name)

    def status(self) -> dict[str, object]:
        thread_alive = bool(self._thread and self._thread.is_alive())
        return {
            "running": thread_alive,
            "thread_name": self.name,
            "poll_seconds": self.poll_seconds,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.scheduler.run_pending_once()
            except Exception as exc:
                logger.exception("Scheduler tick failed: %s", exc)
            self._stop_event.wait(timeout=self.poll_seconds)


class UptimeTracker:
    """Thread-safe uptime ratio helper for API health metrics."""

    def __init__(self) -> None:
        self._started_at = time.time()
        self._total_errors = 0
        self._lock = threading.Lock()

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def uptime_ratio(self) -> float:
        with self._lock:
            elapsed_minutes = max((time.time() - self._started_at) / 60.0, 1.0)
            error_penalty = min(self._total_errors / elapsed_minutes / 10.0, 1.0)
            return max(0.0, 1.0 - error_penalty)
