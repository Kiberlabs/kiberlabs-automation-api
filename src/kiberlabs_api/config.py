"""Environment-backed runtime settings for KiberLabs Automation API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_INSECURE_DEMO_KEYS = frozenset({"kl-dev-demo-key"})


@dataclass(frozen=True)
class GatewaySettings:
    """Immutable gateway configuration loaded from environment variables or test overrides."""

    api_keys: tuple[str, ...]
    db_path: Path
    rate_limit_requests: int = 60
    rate_limit_window_sec: int = 60

    def __post_init__(self) -> None:
        if not self.api_keys or any(not k.strip() for k in self.api_keys):
            raise ValueError("GatewaySettings.api_keys must contain at least one non-empty key")
        if self.rate_limit_requests < 1:
            raise ValueError("rate_limit_requests must be >= 1")
        if self.rate_limit_window_sec < 1:
            raise ValueError("rate_limit_window_sec must be >= 1")

    @classmethod
    def from_env(cls) -> GatewaySettings:
        """Build settings from KIBERLABS_* env vars with fail-fast API key validation."""
        raw_keys = os.getenv("KIBERLABS_API_KEYS", "")
        keys = tuple(dict.fromkeys(k.strip() for k in raw_keys.split(",") if k.strip()))
        if not keys:
            raise ValueError("KIBERLABS_API_KEYS must be set to at least one non-empty key")
        if os.getenv("KIBERLABS_ENV", "").strip().lower() == "production" and any(
            k in _INSECURE_DEMO_KEYS for k in keys
        ):
            raise ValueError("Insecure demo key is not allowed when KIBERLABS_ENV=production")
        db_path = Path(os.getenv("KIBERLABS_DB_PATH") or "data/gateway.db")
        limit = int(os.getenv("KIBERLABS_RATE_LIMIT_RPM", "60"))
        window = int(os.getenv("KIBERLABS_RATE_LIMIT_WINDOW_SEC", "60"))
        return cls(
            api_keys=keys,
            db_path=db_path,
            rate_limit_requests=max(1, limit),
            rate_limit_window_sec=max(1, window),
        )
