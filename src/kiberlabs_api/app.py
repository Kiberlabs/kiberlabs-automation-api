"""FastAPI application factory with API-Key auth, sliding-window rate limiting, and SQLite WAL audit."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

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


def create_app(settings: GatewaySettings | None = None) -> FastAPI:
    """Assemble and configure the production KiberLabs FastAPI gateway.

    Wires request-ID/latency audit middleware, timing-safe X-API-Key authentication,
    per-key sliding-window rate limiting, and all /v1 domain endpoints.
    """
    cfg = settings or GatewaySettings.from_env()
    store = GatewayStore(cfg.db_path)
    limiter = SlidingWindowRateLimiter(cfg.rate_limit_requests, cfg.rate_limit_window_sec)

    application = FastAPI(
        title="KiberLabs B2B Automation API Gateway",
        version="0.1.0",
        description=(
            "Production-ready FastAPI boilerplate exposing deterministic price/margin sieving, "
            "idempotent 24/7 lead webhooks, and zero-hallucination SQLite FTS5 RAG citations."
        ),
    )
    application.state.store = store
    application.state.limiter = limiter
    application.state.settings = cfg

    @application.middleware("http")
    async def audit_middleware(request: Request, call_next: Any) -> Response:
        """Attach X-Request-ID and persist method/path/status/latency telemetry in SQLite WAL."""
        req_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = req_id
        store.log_request(req_id, request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    def require_auth_and_quota(
        response: Response,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> str:
        """Authenticate X-API-Key in constant time and enforce per-key sliding-window quota."""
        if not verify_api_key(x_api_key, cfg.api_keys):
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


app = create_app()
