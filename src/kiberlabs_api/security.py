"""Security primitives: timing-safe API-key auth, sliding-window rate limiter, and SSRF guard."""

from __future__ import annotations

import hmac
import ipaddress
import re
import threading
import time
from collections import deque
from urllib.parse import urlsplit

_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")
_METADATA_NET = ipaddress.ip_network("169.254.0.0/16")
_INTERNAL_HOSTS = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
    }
)
_INTERNAL_SUFFIXES = (".local", ".internal", ".localhost", ".localdomain", ".lan", ".home.arpa", ".corp", ".intranet")
_IPV4_PART_RE = re.compile(r"^(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _parse_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse standard IPv4/IPv6 literals as well as decimal/octal/hex/shorthand IPv4 forms."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    parts = host.split(".")
    if not (1 <= len(parts) <= 4) or not all(_IPV4_PART_RE.match(p) for p in parts):
        return None
    try:
        vals = [
            int(p, 16)
            if p.startswith(("0x", "0X"))
            else (int(p, 8) if p.startswith("0") and len(p) > 1 else int(p, 10))
            for p in parts
        ]
    except ValueError:
        return None
    if len(vals) == 1:
        val = vals[0]
    elif len(vals) == 2 and vals[0] <= 0xFF and vals[1] <= 0xFFFFFF:
        val = (vals[0] << 24) | vals[1]
    elif len(vals) == 3 and vals[0] <= 0xFF and vals[1] <= 0xFF and vals[2] <= 0xFFFF:
        val = (vals[0] << 24) | (vals[1] << 16) | vals[2]
    elif len(vals) == 4 and all(v <= 0xFF for v in vals):
        val = (vals[0] << 24) | (vals[1] << 16) | (vals[2] << 8) | vals[3]
    else:
        return None
    return ipaddress.IPv4Address(val) if 0 <= val <= 0xFFFFFFFF else None


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject private, loopback, CGNAT, metadata, site-local, non-global, and wrapped IPv6 targets."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_blocked_ip(ip.ipv4_mapped)
        if ip.sixtofour is not None and _is_blocked_ip(ip.sixtofour):
            return True
        if ip.teredo is not None and any(_is_blocked_ip(addr) for addr in ip.teredo):
            return True
        if ip.is_site_local:
            return True
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    ):
        return True
    if isinstance(ip, ipaddress.IPv4Address):
        if ip in _CGNAT_NET or ip in _METADATA_NET:
            return True
    return False


def validate_public_http_url(url: str) -> str:
    """Enforce SSRF boundaries by allowing only public HTTP(S) hostnames and IP addresses.

    Rejects non-HTTP schemes, control chars, whitespace, backslashes, invalid ports,
    embedded credentials, localhost, internal suffixes, obfuscated IPv4 literals,
    malformed DNS labels/numeric TLDs, CGNAT, and wrapped/non-public IPv6 addresses.
    """
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")
    if "\\" in url or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc or "\\" in parsed.netloc:
        raise ValueError("Embedded credentials in URL are forbidden")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid port in URL: {url!r}") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Invalid port range in URL: {url!r}")

    host = parsed.hostname.lower()
    if host.endswith("."):
        host = host[:-1]
    if not host or "%" in host or host in _INTERNAL_HOSTS or any(host.endswith(sfx) for sfx in _INTERNAL_SUFFIXES):
        raise ValueError(f"Blocked internal hostname: {host!r}")

    ip = _parse_ip_literal(host)
    if ip is not None:
        if _is_blocked_ip(ip):
            raise ValueError(f"Blocked non-public IP address: {host!r}")
        return url

    labels = host.split(".")
    if (
        len(host) > 253
        or len(labels) < 2
        or labels[-1].isdigit()
        or not all(_DNS_LABEL_RE.match(lbl) for lbl in labels)
    ):
        raise ValueError(f"Invalid public hostname: {host!r}")
    return url


def verify_api_key(candidate: str | None, valid_keys: tuple[str, ...]) -> bool:
    """Verify an incoming X-API-Key header using constant-time HMAC comparison."""
    if not candidate or not candidate.strip() or not valid_keys:
        return False
    matched = False
    for key in valid_keys:
        if key:
            matched |= hmac.compare_digest(candidate, key)
    return matched


class SlidingWindowRateLimiter:
    """Thread-safe bounded-memory sliding-window rate limiter keyed by API credential.

    shortcut: in-memory per-process state; upgrade to Redis sliding window when scaling beyond 1 worker.
    """

    def __init__(self, max_requests: int, window_sec: int, max_buckets: int = 10_000) -> None:
        """Configure request quota (max_requests) over a rolling window_sec interval."""
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.max_buckets = max(1, max_buckets)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def is_blocked(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Check whether key is currently rate-limited without consuming quota."""
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_sec
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket:
                return False, 0
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                del self._buckets[key]
                return False, 0
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_sec - (ts - bucket[0])) + 1)
                return True, retry_after
            return False, 0

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int, int]:
        """Check and record one request for key.

        Returns (allowed, remaining_quota, retry_after_sec).
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_sec
        with self._lock:
            bucket = self._buckets.pop(key, None)
            if bucket is None:
                if len(self._buckets) >= self.max_buckets:
                    expired = [k for k, dq in self._buckets.items() if not dq or dq[-1] <= cutoff]
                    for k in expired:
                        del self._buckets[k]
                    while len(self._buckets) >= self.max_buckets:
                        self._buckets.pop(next(iter(self._buckets)))
                bucket = deque()
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_sec - (ts - bucket[0])) + 1)
                self._buckets[key] = bucket
                return False, 0, retry_after
            bucket.append(ts)
            self._buckets[key] = bucket
            remaining = max(0, self.max_requests - len(bucket))
            return True, remaining, 0
