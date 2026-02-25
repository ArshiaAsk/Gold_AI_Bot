from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditStore:
    """SQLite-backed audit log for predictions, API events, and retraining runs."""

    def __init__(self, db_path: str = "data/mlops_audit.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def from_env(cls) -> "AuditStore":
        return cls(db_path=os.getenv("AUDIT_DB_PATH", "data/mlops_audit.db"))

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.error("Audit store ping failed: %s", exc)
            return False

    def log_prediction(
        self,
        *,
        endpoint: str,
        client_ip: str | None,
        model_version: Optional[int],
        current_price: Optional[float],
        predicted_price: Optional[float],
        latency_ms: Optional[float],
        success: bool,
        error_text: Optional[str] = None,
        confidence_lower: Optional[float] = None,
        confidence_upper: Optional[float] = None,
    ) -> None:
        payload = (
            self._now(),
            endpoint,
            client_ip,
            model_version,
            current_price,
            predicted_price,
            confidence_lower,
            confidence_upper,
            latency_ms,
            int(success),
            error_text,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    timestamp_utc, endpoint, client_ip, model_version, current_price,
                    predicted_price, confidence_lower, confidence_upper, latency_ms, success, error_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def log_training_run(
        self,
        *,
        status: str,
        trigger: str,
        reason: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model_version: Optional[int] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs (
                    timestamp_utc, status, reason, metrics_json, model_version, trigger
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self._now(),
                    status,
                    reason,
                    self._json(metrics),
                    model_version,
                    trigger,
                ),
            )

    def log_event(
        self,
        *,
        event_type: str,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_events (timestamp_utc, event_type, level, message, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._now(), event_type, level, message, self._json(details)),
            )

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    client_ip TEXT,
                    model_version INTEGER,
                    current_price REAL,
                    predicted_price REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    latency_ms REAL,
                    success INTEGER NOT NULL,
                    error_text TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(timestamp_utc);

                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    metrics_json TEXT,
                    model_version INTEGER,
                    trigger TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_training_runs_time ON training_runs(timestamp_utc);

                CREATE TABLE IF NOT EXISTS api_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_api_events_time ON api_events(timestamp_utc);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _json(value: Optional[Dict[str, Any]]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
