"""FastAPI application factory with API-Key auth, sliding-window rate limiting, and SQLite WAL audit."""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kiberlabs_api.config import GatewaySettings
from kiberlabs_api.security import SlidingWindowRateLimiter, verify_api_key
from kiberlabs_api.services import (
    LeadWebhookRequest,
    RagDocumentRequest,
    RagQueryRequest,
    SieveBatchRequest,
    evaluate_sieve_batch,
    handle_lead_webhook,
    index_rag_document,
    query_rag_citations,
)
from kiberlabs_api.storage import GatewayStore

logger = logging.getLogger(__name__)


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    """Assemble and configure the production KiberLabs FastAPI gateway.

    Wires request-ID/latency audit middleware, timing-safe X-API-Key authentication,
    per-key sliding-window rate limiting, and all /v1 domain endpoints.
    """
    cfg = settings or GatewaySettings.from_env()
    store = GatewayStore(cfg.db_path)
    limiter = SlidingWindowRateLimiter(cfg.rate_limit_requests, cfg.rate_limit_window_sec)
    auth_fail_limiter = SlidingWindowRateLimiter(max(10, cfg.rate_limit_requests), cfg.rate_limit_window_sec)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            store.close()

    application = FastAPI(
        title="KiberLabs B2B Automation API Gateway",
        version="0.1.0",
        description=(
            "Production-ready FastAPI boilerplate exposing deterministic price/margin sieving, "
            "idempotent 24/7 lead webhooks, and zero-hallucination SQLite FTS5 RAG citations."
        ),
        lifespan=lifespan,
    )
    application.state.store = store
    application.state.limiter = limiter
    application.state.settings = cfg

    def _sanitize_non_finite(obj: Any) -> Any:
        if isinstance(obj, float) and not math.isfinite(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize_non_finite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_non_finite(v) for v in obj]
        return obj

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": _sanitize_non_finite(jsonable_encoder(exc.errors()))})

    @application.middleware("http")
    async def audit_middleware(request: Request, call_next: Any) -> Response:
        """Attach X-Request-ID and persist method/path/status/latency telemetry in SQLite WAL."""
        req_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            try:
                await asyncio.to_thread(
                    store.log_request, req_id, request.method, request.url.path, status_code, elapsed_ms
                )
            except Exception as exc:
                logger.warning("Failed to persist audit log for %s %s: %s", request.method, request.url.path, exc)

    def require_auth_and_quota(
        request: Request,
        response: Response,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> str:
        """Authenticate X-API-Key in constant time and enforce per-key sliding-window quota."""
        client_ip = request.client.host if request.client else "unknown"
        blocked, retry_fail = auth_fail_limiter.is_blocked(f"authfail:{client_ip}")
        if blocked:
            raise HTTPException(
                status_code=429,
                detail="Too many failed auth attempts",
                headers={"Retry-After": str(retry_fail)},
            )
        if not verify_api_key(x_api_key, cfg.api_keys):
            ok_fail, _, retry_fail = auth_fail_limiter.allow(f"authfail:{client_ip}")
            if not ok_fail:
                raise HTTPException(
                    status_code=429,
                    detail="Too many failed auth attempts",
                    headers={"Retry-After": str(retry_fail)},
                )
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
        assert x_api_key is not None
        allowed, remaining, retry_after = limiter.allow(x_api_key)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        return x_api_key

    @application.get("/health", tags=["system"])
    def healthcheck() -> dict[str, str]:
        """Unauthenticated liveness probe verifying SQLite WAL availability."""
        return {"status": "ok", "journal_mode": store.journal_mode(), "version": "0.1.0"}

    @application.post("/v1/sieve/evaluate", tags=["sieve"])
    def post_sieve_evaluate(
        payload: SieveBatchRequest,
        _: str = Depends(require_auth_and_quota),
    ) -> dict[str, object]:
        """Evaluate supplier SKUs through SSRF and net-margin unit-economics gates."""
        return evaluate_sieve_batch(payload)

    @application.post("/v1/leads/webhook", tags=["leads"])
    def post_lead_webhook(
        payload: LeadWebhookRequest,
        _: str = Depends(require_auth_and_quota),
    ) -> dict[str, object]:
        """Ingest WhatsApp/Telegram/Web webhooks with atomic SQLite WAL deduplication."""
        return handle_lead_webhook(payload, store)

    @application.post("/v1/rag/documents", tags=["rag"])
    def post_rag_document(
        payload: RagDocumentRequest,
        _: str = Depends(require_auth_and_quota),
    ) -> dict[str, object]:
        """Index a structured regulatory/policy document into SQLite FTS5 by Article/Section."""
        return index_rag_document(payload, store)

    @application.post("/v1/rag/query", tags=["rag"])
    def post_rag_query(
        payload: RagQueryRequest,
        _: str = Depends(require_auth_and_quota),
    ) -> dict[str, object]:
        """Retrieve BM25-ranked citations or deterministically abstain on insufficient evidence."""
        return query_rag_citations(payload, store)

    @application.get("/v1/audit/recent", tags=["system"])
    def get_recent_audit(
        limit: int = 20,
        _: str = Depends(require_auth_and_quota),
    ) -> dict[str, object]:
        """Return recent authenticated/public API request telemetry from SQLite WAL."""
        bounded = max(1, min(limit, 100))
        return {"items": store.recent_audit(limit=bounded)}

    return application


def __getattr__(name: str) -> Any:
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
