from __future__ import annotations

import os
import signal
import threading
import time

from config.logging import setup_logging
from config.settings import Config
from src.mlops.api_integration import initialize_mlops


def run_worker() -> None:
    setup_logging(base_log_dir="logs", level=os.getenv("LOG_LEVEL", "INFO"))

    poll_seconds = int(os.getenv("MLOPS_SCHEDULER_POLL_SECONDS", "60"))
    prometheus_port = int(os.getenv("PROMETHEUS_WORKER_PORT", "8001"))

    integration = initialize_mlops(
        config=Config(),
        scheduler_poll_seconds=poll_seconds,
        start_prometheus_http_server=True,
        prometheus_port=prometheus_port,
    )

    stop_event = threading.Event()

    def _graceful_shutdown(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1)
    finally:
        integration.stop()


if __name__ == "__main__":
    run_worker()
