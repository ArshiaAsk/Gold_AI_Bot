from __future__ import annotations

import logging
import os
import smtplib
import time
from dataclasses import dataclass, field
from email.message import EmailMessage

import requests

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

        self._send_slack(payload)
        self._send_telegram(payload)
        self._send_email(subject=f"{self.service_name}::{title}", body=payload)

    @staticmethod
    def _send_slack(payload: str) -> None:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return
        try:
            response = requests.post(webhook_url, json={"text": payload}, timeout=5)
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Slack alert failed: %s", exc)

    @staticmethod
    def _send_telegram(payload: str) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            response = requests.post(
                url,
                json={"chat_id": chat_id, "text": payload},
                timeout=5,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Telegram alert failed: %s", exc)

    @staticmethod
    def _send_email(subject: str, body: str) -> None:
        smtp_host = os.getenv("ALERT_SMTP_HOST")
        smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        smtp_user = os.getenv("ALERT_SMTP_USER")
        smtp_pass = os.getenv("ALERT_SMTP_PASSWORD")
        from_addr = os.getenv("ALERT_EMAIL_FROM")
        to_addr = os.getenv("ALERT_EMAIL_TO")
        if not (smtp_host and smtp_user and smtp_pass and from_addr and to_addr):
            return

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(body)

        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
                smtp.starttls()
                smtp.login(smtp_user, smtp_pass)
                smtp.send_message(msg)
        except Exception as exc:
            logger.warning("Email alert failed: %s", exc)
