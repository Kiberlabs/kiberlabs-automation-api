"""Security primitives: timing-safe API-key auth, sliding-window rate limiter, and SSRF guard."""

from __future__ import annotations

import hmac
import ipaddress
import threading
import time
from collections import deque
from urllib.parse import urlsplit


def validate_public_http_url(url: str) -> str:
    """Enforce SSRF boundaries by allowing only public HTTP(S) hostnames and IP addresses.

    Rejects non-HTTP schemes, localhost, internal suffixes (.local/.internal), and private,
    loopback, link-local, multicast, or reserved IP literals.
    """
    parsed = urlsplit(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")

    host = parsed.hostname.lower().rstrip(".")
    if host in ("localhost", "ip6-localhost") or host.endswith((".local", ".internal", ".localhost")):
        raise ValueError(f"Blocked internal hostname: {host!r}")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url.strip()

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(f"Blocked non-public IP address: {host!r}")
    return url.strip()


def verify_api_key(candidate: str | None, valid_keys: tuple[str, ...]) -> bool:
    """Verify an incoming X-API-Key header using constant-time HMAC comparison."""
    if not candidate:
        return False
    return any(hmac.compare_digest(candidate, key) for key in valid_keys)


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by API credential.

    shortcut: in-memory per-process state; upgrade to Redis sliding window when scaling beyond 1 worker.
    """

    def __init__(self, max_requests: int, window_sec: int) -> None:
        """Configure request quota (max_requests) over a rolling window_sec interval."""
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int, int]:
        """Check and record one request for key.

        Returns (allowed, remaining_quota, retry_after_sec).
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_sec
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_sec - (ts - bucket[0])) + 1)
                return False, 0, retry_after
            bucket.append(ts)
            remaining = max(0, self.max_requests - len(bucket))
            return True, remaining, 0
