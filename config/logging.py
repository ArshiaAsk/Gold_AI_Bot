from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    base_log_dir: str = "logs",
    level: str | int = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure console + rotating file handlers for API/MLOps/training logs."""
    if isinstance(level, str):
        log_level = logging._nameToLevel.get(level.upper(), logging.INFO)
    else:
        log_level = int(level)
    Path(base_log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers when app reloads.
    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(fmt)

    api_file = RotatingFileHandler(
        os.path.join(base_log_dir, "api.log"), maxBytes=max_bytes, backupCount=backup_count
    )
    api_file.setLevel(log_level)
    api_file.setFormatter(fmt)

    mlops_file = RotatingFileHandler(
        os.path.join(base_log_dir, "mlops.log"), maxBytes=max_bytes, backupCount=backup_count
    )
    mlops_file.setLevel(log_level)
    mlops_file.setFormatter(fmt)

    training_file = RotatingFileHandler(
        os.path.join(base_log_dir, "training.log"), maxBytes=max_bytes, backupCount=backup_count
    )
    training_file.setLevel(log_level)
    training_file.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(api_file)

    logging.getLogger("src.mlops").addHandler(mlops_file)
    logging.getLogger("src.mlops").propagate = True

    logging.getLogger("src.train_pipeline").addHandler(training_file)
    logging.getLogger("src.train_pipeline").propagate = True
