from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = os.getenv("API_KEY_HEADER_NAME", "X-API-Key")
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


AUTH_REQUIRED = _as_bool(os.getenv("API_KEY_REQUIRED"), True)
API_KEY_VALUE = os.getenv("API_KEY")

RATE_LIMIT_ENABLED = _as_bool(os.getenv("RATE_LIMIT_ENABLED"), True)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


@dataclass
class RateLimiter:
    requests: int
    window_seconds: int
    _bucket: dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._bucket[key]
            while q and (now - q[0]) > self.window_seconds:
                q.popleft()
            if len(q) >= self.requests:
                return False
            q.append(now)
            return True

    def retry_after_seconds(self, key: str) -> int:
        with self._lock:
            q = self._bucket.get(key)
            if not q:
                return self.window_seconds
            remaining = self.window_seconds - (time.time() - q[0])
            return max(1, int(remaining))


rate_limiter = RateLimiter(
    requests=max(1, RATE_LIMIT_REQUESTS),
    window_seconds=max(1, RATE_LIMIT_WINDOW_SECONDS),
)


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if not AUTH_REQUIRED:
        return
    if not API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key authentication is enabled but API_KEY is not configured",
        )
    if not api_key or api_key != API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
