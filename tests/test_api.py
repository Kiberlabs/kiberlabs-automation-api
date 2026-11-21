"""End-to-end integration tests for KiberLabs Automation API Gateway."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from kiberlabs_api.app import create_app
from kiberlabs_api.config import GatewaySettings


def _build_client(tmp_path: Path, rpm: int = 50) -> TestClient:
    settings = GatewaySettings(
        api_keys=("kl-test-secret-key",),
        db_path=tmp_path / "test_gateway.db",
        rate_limit_requests=rpm,
        rate_limit_window_sec=60,
    )
    return TestClient(create_app(settings))


def test_healthcheck_and_sqlite_wal_mode(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["journal_mode"] == "wal"
    assert "X-Request-ID" in resp.headers


def test_missing_or_invalid_api_key_returns_401(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    assert client.get("/v1/audit/recent").status_code == 401
    assert client.get("/v1/audit/recent", headers={"X-API-Key": "wrong-key"}).status_code == 401


def test_sliding_window_rate_limiter_enforces_429(tmp_path: Path) -> None:
    client = _build_client(tmp_path, rpm=2)
    headers = {"X-API-Key": "kl-test-secret-key"}
    r1 = client.get("/v1/audit/recent", headers=headers)
    r2 = client.get("/v1/audit/recent", headers=headers)
    r3 = client.get("/v1/audit/recent", headers=headers)
    assert r1.status_code == 200 and r1.headers["X-RateLimit-Remaining"] == "1"
    assert r2.status_code == 200 and r2.headers["X-RateLimit-Remaining"] == "0"
    assert r3.status_code == 429 and "Retry-After" in r3.headers


def test_sieve_evaluate_and_ssrf_rejection(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    headers = {"X-API-Key": "kl-test-secret-key"}

    ok_resp = client.post(
        "/v1/sieve/evaluate",
        headers=headers,
        json={
            "items": [
                {
                    "sku": "KB-DOCK-01",
                    "title": "Thunderbolt 4 Dock",
                    "supplier_url": "https://supplier.example.com/dock",
                    "supplier_price_eur": 85.0,
                    "target_resale_eur": 159.0,
                    "shipping_cost_eur": 6.0,
                }
            ]
        },
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["approved_count"] == 1

    ssrf_resp = client.post(
        "/v1/sieve/evaluate",
        headers=headers,
        json={
            "items": [
                {
                    "sku": "KB-BAD-SSRF",
                    "title": "Internal Metadata SSRF",
                    "supplier_url": "http://169.254.169.254/latest/meta-data",
                    "supplier_price_eur": 10.0,
                    "target_resale_eur": 50.0,
                }
            ]
        },
    )
    assert ssrf_resp.status_code == 422


def test_lead_webhook_atomic_idempotency_and_spam(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    headers = {"X-API-Key": "kl-test-secret-key"}

    payload = {
        "channel": "whatsapp",
        "message_id": "wamid-001",
        "sender_id": "client-es-01",
        "text": "Hola, necesitamos bot de WhatsApp con presupuesto de 1.200€",
    }
    first = client.post("/v1/leads/webhook", headers=headers, json=payload)
    dup = client.post("/v1/leads/webhook", headers=headers, json=payload)
    assert first.status_code == 200
    assert first.json()["duplicate"] is False
    assert first.json()["status"] == "HANDOFF_READY"
    assert first.json()["estimated_budget_eur"] == 1200.0
    assert dup.status_code == 200
    assert dup.json()["duplicate"] is True
    assert dup.json()["status"] == "DUPLICATE_IGNORED"

    spam = client.post(
        "/v1/leads/webhook",
        headers=headers,
        json={
            "channel": "telegram",
            "message_id": "tg-999",
            "sender_id": "spammer",
            "text": "Free crypto airdrop click now!",
        },
    )
    assert spam.status_code == 200
    assert spam.json()["status"] == "SPAM_BLOCKED"


def test_rag_index_grounded_query_and_abstention(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    headers = {"X-API-Key": "kl-test-secret-key"}

    ing = client.post(
        "/v1/rag/documents",
        headers=headers,
        json={
            "doc_id": "REG-01",
            "title": "B2B Compliance Policy",
            "content": (
                "Article 4: Breach Notification\n"
                "Processors must notify controllers within 72 hours of a confirmed data breach."
            ),
        },
    )
    assert ing.status_code == 200
    assert ing.json()["chunks_indexed"] == 1

    hit = client.post(
        "/v1/rag/query",
        headers=headers,
        json={"query": "data breach notification 72 hours"},
    )
    assert hit.status_code == 200
    assert hit.json()["status"] == "GROUNDED"
    assert hit.json()["citations"][0]["doc_id"] == "REG-01"

    abstain = client.post(
        "/v1/rag/query",
        headers=headers,
        json={"query": "quantum aerospace thruster propellant formula"},
    )
    assert abstain.status_code == 200
    assert abstain.json()["status"] == "INSUFFICIENT_EVIDENCE"
    assert abstain.json()["citations"] == []
