"""Environment-backed runtime settings for KiberLabs Automation API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GatewaySettings:
    """Immutable gateway configuration loaded from environment variables or test overrides."""

    api_keys: tuple[str, ...]
    db_path: Path
    rate_limit_requests: int = 60
    rate_limit_window_sec: int = 60

    @classmethod
    def from_env(cls) -> GatewaySettings:
        """Build settings from KIBERLABS_* env vars with safe local development defaults."""
        raw_keys = os.getenv("KIBERLABS_API_KEYS", "kl-dev-demo-key")
        keys = tuple(k.strip() for k in raw_keys.split(",") if k.strip())
        db_path = Path(os.getenv("KIBERLABS_DB_PATH", "data/gateway.db"))
        limit = int(os.getenv("KIBERLABS_RATE_LIMIT_RPM", "60"))
        window = int(os.getenv("KIBERLABS_RATE_LIMIT_WINDOW_SEC", "60"))
        return cls(
            api_keys=keys,
            db_path=db_path,
            rate_limit_requests=max(1, limit),
            rate_limit_window_sec=max(1, window),
        )
