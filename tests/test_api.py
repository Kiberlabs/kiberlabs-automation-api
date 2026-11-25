"""End-to-end integration tests for KiberLabs Automation API Gateway."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import random
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kiberlabs_api.app as app_module
from kiberlabs_api.app import create_app
from kiberlabs_api.config import GatewaySettings
from kiberlabs_api.security import SlidingWindowRateLimiter, validate_public_http_url
from kiberlabs_api.services import (
    SieveBatchRequest,
    _extract_budget_eur,
    classify_lead_service,
    evaluate_sieve_batch,
)
from kiberlabs_api.storage import GatewayStore


def _build_client(tmp_path: Path, rpm: int = 50) -> TestClient:
    settings = GatewaySettings(
        api_keys=("kl-test-secret-key",),
        db_path=tmp_path / "test_gateway.db",
        rate_limit_requests=rpm,
        rate_limit_window_sec=60,
    )
    return TestClient(create_app(settings))


def test_healthcheck_and_sqlite_wal_mode(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["journal_mode"] == "wal"
        assert "X-Request-ID" in resp.headers


def test_missing_or_invalid_api_key_returns_401(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        assert client.get("/v1/audit/recent").status_code == 401
        assert client.get("/v1/audit/recent", headers={"X-API-Key": "wrong-key"}).status_code == 401


def test_sliding_window_rate_limiter_enforces_429(tmp_path: Path) -> None:
    with _build_client(tmp_path, rpm=2) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        r1 = client.get("/v1/audit/recent", headers=headers)
        r2 = client.get("/v1/audit/recent", headers=headers)
        r3 = client.get("/v1/audit/recent", headers=headers)
        assert r1.status_code == 200 and r1.headers["X-RateLimit-Remaining"] == "1"
        assert r2.status_code == 200 and r2.headers["X-RateLimit-Remaining"] == "0"
        assert r3.status_code == 429 and "Retry-After" in r3.headers


def test_sieve_evaluate_and_ssrf_rejection(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
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
    with _build_client(tmp_path) as client:
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
    with _build_client(tmp_path) as client:
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


def test_multithreaded_concurrency_no_leaks_or_500s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened_conns: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened_conns.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)

    settings = GatewaySettings(
        api_keys=("kl-test-secret-key",),
        db_path=tmp_path / "mt_gateway.db",
        rate_limit_requests=200,
        rate_limit_window_sec=60,
    )
    with TestClient(create_app(settings)) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}

        seed_rag = client.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "doc_id": "MT-REG",
                "title": "Concurrency Policy",
                "content": "Article 1: SLA\nAutomated gateway webhooks handle idempotent delivery within 72 hours.",
            },
        )
        assert seed_rag.status_code == 200

        def task(idx: int) -> tuple[str, int, dict[str, object]]:
            kind = idx % 5
            if kind == 0:
                r = client.get("/health")
                return ("health", r.status_code, {"rid": r.headers.get("X-Request-ID")})
            if kind == 1:
                r = client.post(
                    "/v1/sieve/evaluate",
                    headers=headers,
                    json={
                        "items": [
                            {
                                "sku": f"MT-SKU-{idx}",
                                "title": "EU Dock",
                                "supplier_url": "https://supplier.example.com/item",
                                "supplier_price_eur": 40.0,
                                "target_resale_eur": 95.0,
                                "shipping_cost_eur": 5.0,
                            }
                        ]
                    },
                )
                return ("sieve", r.status_code, {"rid": r.headers.get("X-Request-ID")})
            if kind == 2:
                r = client.post(
                    "/v1/leads/webhook",
                    headers=headers,
                    json={
                        "channel": "whatsapp",
                        "message_id": "shared-mt-wamid",
                        "sender_id": "lead-mt",
                        "text": "Presupuesto 1.500 EUR",
                    },
                )
                return ("webhook", r.status_code, {"rid": r.headers.get("X-Request-ID"), "body": r.json()})
            if kind == 3:
                r = client.post(
                    "/v1/rag/query",
                    headers=headers,
                    json={"query": "idempotent delivery 72 hours"},
                )
                return ("rag", r.status_code, {"rid": r.headers.get("X-Request-ID")})
            r = client.get("/v1/audit/recent?limit=10", headers=headers)
            return ("audit", r.status_code, {"rid": r.headers.get("X-Request-ID")})

        with ThreadPoolExecutor(max_workers=16) as pool:
            outcomes = list(pool.map(task, range(50)))

        assert all(code == 200 for _, code, _ in outcomes)
        assert all(bool(meta["rid"]) for _, _, meta in outcomes)
        webhook_bodies = [meta["body"] for kind, _, meta in outcomes if kind == "webhook"]
        assert len(webhook_bodies) == 10
        assert sum(1 for b in webhook_bodies if b["duplicate"] is False) == 1  # type: ignore[index]

    assert len(opened_conns) > 0
    for conn in opened_conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_audit_middleware_logs_500_on_unhandled_exception(tmp_path: Path) -> None:
    settings = GatewaySettings(
        api_keys=("kl-test-secret-key",),
        db_path=tmp_path / "err_gateway.db",
    )
    app = create_app(settings)

    @app.get("/boom-test")
    def boom_route() -> None:
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/boom-test")
        assert resp.status_code == 500

        audit_resp = client.get("/v1/audit/recent?limit=10", headers={"X-API-Key": "kl-test-secret-key"})
        assert audit_resp.status_code == 200
        items = audit_resp.json()["items"]
        assert any(item["path"] == "/boom-test" and item["status_code"] == 500 for item in items)


def test_lazy_app_import_and_api_key_config_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    importlib.reload(app_module)
    assert not (tmp_path / "data" / "gateway.db").exists()

    monkeypatch.delenv("KIBERLABS_API_KEYS", raising=False)
    monkeypatch.delenv("KIBERLABS_ENV", raising=False)
    with pytest.raises(ValueError, match="KIBERLABS_API_KEYS"):
        GatewaySettings.from_env()

    monkeypatch.setenv("KIBERLABS_API_KEYS", "   ,  ")
    with pytest.raises(ValueError, match="KIBERLABS_API_KEYS"):
        GatewaySettings.from_env()

    monkeypatch.setenv("KIBERLABS_API_KEYS", "kl-dev-demo-key")
    dev_cfg = GatewaySettings.from_env()
    assert dev_cfg.api_keys == ("kl-dev-demo-key",)

    monkeypatch.setenv("KIBERLABS_ENV", "production")
    with pytest.raises(ValueError, match="production"):
        GatewaySettings.from_env()


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0177.0.0.1/",
        "http://127.1/",
        "http://100.64.0.1/",
        "http://user:pass@example.com/",
        "http://[::ffff:127.0.0.1]/",
        "http://[2002:7f00:1::]/",
        "http://[2001:0000:7f00:0001::]/",
        "http://example.com\t/",
    ],
)
def test_sieve_rejects_non_finite_floats_and_obfuscated_ssrf(tmp_path: Path, bad_url: str) -> None:
    with _build_client(tmp_path) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}

        ssrf_resp = client.post(
            "/v1/sieve/evaluate",
            headers=headers,
            json={
                "items": [
                    {
                        "sku": "SSRF-1",
                        "title": "Obfuscated SSRF",
                        "supplier_url": bad_url,
                        "supplier_price_eur": 10.0,
                        "target_resale_eur": 50.0,
                    }
                ]
            },
        )
        assert ssrf_resp.status_code == 422

        raw_inf_body = (
            '{"items":[{"sku":"INF-1","title":"Inf","supplier_url":"https://supplier.example.com/1",'
            '"supplier_price_eur":10.0,"target_resale_eur":Infinity}]}'
        )
        inf_resp = client.post(
            "/v1/sieve/evaluate",
            headers={**headers, "Content-Type": "application/json"},
            content=raw_inf_body,
        )
        assert inf_resp.status_code == 422


@pytest.mark.parametrize(
    ("text", "expected_status", "expected_budget"),
    [
        ("Our budget is $1000000 for rollout", "NEEDS_BUDGET", None),
        ("Our budget is 1000000 EUR for rollout", "NEEDS_BUDGET", None),
        ("Approved budget is 850.50 EUR", "HANDOFF_READY", 850.5),
        ("Presupuesto de 1.200,50 EUR para el bot", "HANDOFF_READY", 1200.5),
        ("We operate in 500 europe offices", "NEEDS_BUDGET", None),
    ],
)
def test_lead_budget_boundary_and_decimal_regression(
    tmp_path: Path,
    text: str,
    expected_status: str,
    expected_budget: float | None,
) -> None:
    with _build_client(tmp_path) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        resp = client.post(
            "/v1/leads/webhook",
            headers=headers,
            json={
                "channel": "web",
                "message_id": f"msg-{abs(hash(text))}",
                "sender_id": "lead-check",
                "text": text,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == expected_status
        assert body["estimated_budget_eur"] == expected_budget


def test_rag_sanitization_chunking_and_limit_before_overlap(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}

        bad_doc = client.post(
            "/v1/rag/documents",
            headers=headers,
            json={"doc_id": "BAD-01", "title": "Valid Title", "content": "   \x00   short   "},
        )
        assert bad_doc.status_code == 422

        long_words = " ".join(["compliance"] * 400)  # > 3000 chars, no headings
        long_doc = client.post(
            "/v1/rag/documents",
            headers=headers,
            json={"doc_id": "LONG-01", "title": "Monolithic Doc", "content": long_words},
        )
        assert long_doc.status_code == 200
        assert long_doc.json()["chunks_indexed"] >= 3

        for i in range(4):
            client.post(
                "/v1/rag/documents",
                headers=headers,
                json={
                    "doc_id": f"NOISE-{i}",
                    "title": f"Narwhal Noise {i}",
                    "content": f"Article 1: Noise\nNarwhal telemetry narwhal marker {i} repeated narwhal text.",
                },
            )
        client.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "doc_id": "TARGET-DOC",
                "title": "Audit SLA",
                "content": "Article 2: Escrow\nNarwhal encrypted escrow compliance verification mandatory.",
            },
        )

        q_resp = client.post(
            "/v1/rag/query",
            headers=headers,
            json={
                "query": "narwhal encrypted escrow compliance verification",
                "top_k": 2,
                "min_token_overlap": 0.5,
            },
        )
        assert q_resp.status_code == 200
        q_body = q_resp.json()
        assert q_body["status"] == "GROUNDED"
        assert 1 <= len(q_body["citations"]) <= 2
        assert q_body["citations"][0]["doc_id"] == "TARGET-DOC"


def test_rate_limiter_bounded_buckets_and_failed_auth_throttle(tmp_path: Path) -> None:
    limiter = SlidingWindowRateLimiter(max_requests=5, window_sec=60, max_buckets=3)
    for i in range(10):
        limiter.allow(f"key-{i}", now=100.0 + i)
    assert len(limiter._buckets) <= 3

    with _build_client(tmp_path, rpm=2) as client:
        statuses = [
            client.get("/v1/audit/recent", headers={"X-API-Key": f"bad-key-{i}"})
            for i in range(15)
        ]
        codes = [r.status_code for r in statuses]
        assert 401 in codes
        assert 429 in codes
        assert "Retry-After" in statuses[-1].headers


def test_rag_stopword_nonsense_and_randomized_query_fuzzing(tmp_path: Path) -> None:
    rng = random.Random(20261123)
    stopwords = [
        "to", "in", "of", "on", "at", "by", "an", "is", "as",
        "de", "en", "el", "la", "es", "un", "al",
        "the", "and", "for", "with", "los", "las", "para", "sobre",
    ]
    nonces = ["zxqcake", "pan", "cake", "zumbopan", "flurbos"]
    corpus_terms = ["breach", "72", "hours", "gdpr", "facturas", "escrow"]
    fts_ops = ['"', "*", "^", "(", ")", "OR", "AND", "NOT", "NEAR", "\x00", "á", "é", "ñ"]

    with _build_client(tmp_path, rpm=500) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        ing = client.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "doc_id": "FUZZ-RAG-01",
                "title": "GDPR Breach & Facturas Policy",
                "content": (
                    "Article 1: Breach SLA\n"
                    "Under GDPR processors must notify controllers within 72 hours of a breach.\n\n"
                    "Article 2: Facturas\n"
                    "Conservacion obligatoria de facturas electronicas y escrow."
                ),
            },
        )
        assert ing.status_code == 200

        for fixed_q in ("cake in to is", "pan en el de", "to in of", "en el de la es un al"):
            resp = client.post("/v1/rag/query", headers=headers, json={"query": fixed_q})
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "INSUFFICIENT_EVIDENCE"
            assert body["citations"] == []
            assert body["verified"] is True

        for real_q in ("gdpr", "breach in to is", "facturas en el de"):
            resp = client.post("/v1/rag/query", headers=headers, json={"query": real_q})
            assert resp.status_code == 200
            assert resp.json()["status"] == "GROUNDED"

        for _ in range(60):
            picked = rng.choices(stopwords, k=rng.randint(1, 5)) + rng.choices(nonces, k=rng.randint(0, 2))
            rng.shuffle(picked)
            q = " ".join(picked)
            if len(q.strip()) < 2:
                continue
            resp = client.post("/v1/rag/query", headers=headers, json={"query": q})
            assert resp.status_code == 200
            assert resp.json()["status"] == "INSUFFICIENT_EVIDENCE"
            assert resp.json()["citations"] == []

        for _ in range(60):
            parts = rng.choices(stopwords + corpus_terms + nonces + fts_ops, k=rng.randint(1, 8))
            q = " ".join(parts)[:500]
            top_k = rng.randint(1, 5)
            resp = client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query": q, "top_k": top_k},
            )
            assert resp.status_code in (200, 422)
            if resp.status_code == 200:
                body = resp.json()
                assert body["verified"] is True
                assert len(body["citations"]) <= top_k


def test_rag_33k_chunk_reingestion_and_content_max_length(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}

        too_long = client.post(
            "/v1/rag/documents",
            headers=headers,
            json={"doc_id": "TOO-LONG", "title": "Oversized Doc", "content": "A" * 200_001},
        )
        assert too_long.status_code == 422

        store = client.app.state.store  # type: ignore[attr-defined]
        seed_rows = [
            (f"BIG-API-DOC::c{i:05d}", "Huge Legacy", f"Sec {i}", f"Legacyapitoken row {i}", f"{i:016x}")
            for i in range(1, 33_501)
        ]
        assert store.upsert_rag_chunks("BIG-API-DOC", seed_rows) == 33_500

        replace_resp = client.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "doc_id": "BIG-API-DOC",
                "title": "Compact Replacement",
                "content": "Article 1: Zero Trust\nCryptographic attestation verification replaces legacy tokens.",
            },
        )
        assert replace_resp.status_code == 200
        assert replace_resp.json()["chunks_indexed"] == 1

        with store._connect() as conn:
            cnt_chunks = conn.execute(
                "SELECT COUNT(*) FROM rag_chunks WHERE doc_id = ?", ("BIG-API-DOC",)
            ).fetchone()[0]
            cnt_fts = conn.execute(
                "SELECT COUNT(*) FROM rag_chunks_fts WHERE chunk_id LIKE 'BIG-API-DOC::%'"
            ).fetchone()[0]
        assert cnt_chunks == 1
        assert cnt_fts == 1

        old_q = client.post("/v1/rag/query", headers=headers, json={"query": "legacyapitoken"})
        assert old_q.status_code == 200
        assert old_q.json()["status"] == "INSUFFICIENT_EVIDENCE"

        new_q = client.post("/v1/rag/query", headers=headers, json={"query": "cryptographic attestation"})
        assert new_q.status_code == 200
        assert new_q.json()["status"] == "GROUNDED"


def test_fuzz_sieve_malformed_and_obfuscated_urls(tmp_path: Path) -> None:
    rng = random.Random(20261124)
    fixed_invalid = [
        "http://evil-.com/",
        "http://-evil.com/",
        "http://evil..com/",
        "http://example.com../",
        "http://example.123/",
        "http://127.0.0.08/",
        "http://08.0.0.1/",
        "http://127.0.0.256/",
        "http://[fec0::1]/",
        "http://ip6-localhost/",
        "http://foo.localdomain/",
        "http://example.com/foo\\bar",
        "http://127.0.0.1\\a.com/",
        "http://example.com:99999/",
        "http://example.com:0/",
        "  https://example.com/  ",
        f"https://{'a' * 64}.example.com/",
        f"https://example.com/{'x' * 2040}",
    ]
    generators = [
        lambda: f"http://{rng.choice(['a', 'evil', 'sub1'])}-.example.com/",
        lambda: f"http://-{rng.choice(['a', 'evil', 'sub1'])}.example.com/",
        lambda: f"http://a..{rng.choice(['b', 'c'])}.com/",
        lambda: f"http://example.{rng.randint(100, 999)}/",
        lambda: f"http://127.0.0.0{rng.randint(8, 9)}/",
        lambda: f"http://sub.{rng.choice(['localdomain', 'local', 'internal', 'lan', 'corp'])}/",
        lambda: f"http://example.com:{rng.choice([0, 65536, 99999])}/",
        lambda: f"http://example.com/path\\{rng.randint(1, 99)}",
        lambda: f"  https://sub{rng.randint(1, 50)}.example.com/  ",
    ]
    invalid_urls = fixed_invalid + [rng.choice(generators)() for _ in range(90)]

    valid_urls = [
        "https://example.com./",
        "https://sub-1.supplier.example.com:8443/item?q=1&b=2",
        "http://8.8.8.8/dns",
    ] + [
        f"https://shop-{i}.supplier-eu.example.com:{rng.choice([80, 443, 8080])}/sku/{i}?ref=ok"
        for i in range(27)
    ]

    for bad in invalid_urls:
        with pytest.raises(ValueError):
            validate_public_http_url(bad)

    for good in valid_urls:
        assert validate_public_http_url(good) == good

    with _build_client(tmp_path, rpm=500) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        for idx, bad in enumerate(invalid_urls[:30]):
            resp = client.post(
                "/v1/sieve/evaluate",
                headers=headers,
                json={
                    "items": [
                        {
                            "sku": f"BAD-{idx}",
                            "title": "Bad URL",
                            "supplier_url": bad,
                            "supplier_price_eur": 10.0,
                            "target_resale_eur": 50.0,
                        }
                    ]
                },
            )
            assert resp.status_code == 422

        ok_resp = client.post(
            "/v1/sieve/evaluate",
            headers=headers,
            json={
                "items": [
                    {
                        "sku": f"OK-{idx}",
                        "title": "Valid URL",
                        "supplier_url": good,
                        "supplier_price_eur": 20.0,
                        "target_resale_eur": 60.0,
                    }
                    for idx, good in enumerate(valid_urls)
                ]
            },
        )
        assert ok_resp.status_code == 200
        assert ok_resp.json()["processed"] == len(valid_urls)


def test_fuzz_sieve_bool_nan_overflow_and_formula_parity(tmp_path: Path) -> None:
    with _build_client(tmp_path, rpm=500) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        base_item = {
            "sku": "BOOL-1",
            "title": "Bool Guard",
            "supplier_url": "https://supplier.example.com/item",
            "supplier_price_eur": 20.0,
            "target_resale_eur": 60.0,
        }
        for field in ("supplier_price_eur", "target_resale_eur", "shipping_cost_eur", "delivery_days", "rating"):
            bad_item = {**base_item, field: True}
            resp = client.post("/v1/sieve/evaluate", headers=headers, json={"items": [bad_item]})
            assert resp.status_code == 422

        for batch_field in ("marketplace_fee_pct", "return_reserve_pct", "min_net_margin_pct", "min_net_profit_eur"):
            resp = client.post(
                "/v1/sieve/evaluate",
                headers=headers,
                json={"items": [base_item], batch_field: False},
            )
            assert resp.status_code == 422

        for q_field in ("top_k", "min_token_overlap"):
            resp = client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query": "breach notification", q_field: True},
            )
            assert resp.status_code == 422

        for fee, reserve in ((0.6, 0.4), (0.7, 0.4), (0.79, 0.21)):
            resp = client.post(
                "/v1/sieve/evaluate",
                headers=headers,
                json={"items": [base_item], "marketplace_fee_pct": fee, "return_reserve_pct": reserve},
            )
            assert resp.status_code == 422

        for raw_bad in (
            '{"items":[{"sku":"NAN","title":"NaN","supplier_url":"https://supplier.example.com/1","supplier_price_eur":10.0,"target_resale_eur":NaN}]}',
            '{"items":[{"sku":"NINF","title":"-Inf","supplier_url":"https://supplier.example.com/1","supplier_price_eur":10.0,"target_resale_eur":-Infinity}]}',
            json.dumps({"items": [{**base_item, "target_resale_eur": 1e308}]}),
            json.dumps({"items": [{**base_item, "supplier_price_eur": 10.0, "target_resale_eur": 1e-320}]}),
        ):
            resp = client.post(
                "/v1/sieve/evaluate",
                headers={**headers, "Content-Type": "application/json"},
                content=raw_bad,
            )
            assert resp.status_code == 422

    # 200-trial formula & sorting parity against #1 / engines.js
    rng = random.Random(42)
    for idx in range(200):
        fee_pct = round(rng.uniform(0.0, 0.25), 3)
        reserve_pct = round(rng.uniform(0.0, 0.15), 3)
        min_profit = round(rng.uniform(2.0, 15.0), 2)
        min_margin = round(rng.uniform(0.10, 0.30), 2)
        items_payload = []
        for j in range(4):
            items_payload.append(
                {
                    "sku": f"P-{idx}-{j}",
                    "title": f"Item {j}",
                    "supplier_url": "https://supplier.example.com/p",
                    "supplier_price_eur": round(rng.uniform(5.0, 120.0), 2),
                    "target_resale_eur": round(rng.uniform(15.0, 220.0), 2),
                    "shipping_cost_eur": round(rng.uniform(0.0, 15.0), 2),
                    "in_stock": rng.random() > 0.1,
                    "ships_from_eu": rng.random() > 0.1,
                    "delivery_days": rng.randint(1, 8),
                    "rating": round(rng.uniform(3.5, 5.0), 2),
                }
            )
        req = SieveBatchRequest.model_validate(
            {
                "items": items_payload,
                "marketplace_fee_pct": fee_pct,
                "return_reserve_pct": reserve_pct,
                "min_net_margin_pct": min_margin,
                "min_net_profit_eur": min_profit,
            }
        )
        res = evaluate_sieve_batch(req)
        verdicts = res["verdicts"]
        assert isinstance(verdicts, list)

        # Verify sorting invariants: (approved, score, net_profit_eur) descending
        sort_keys = [(bool(v["approved"]), float(v["score"]), float(v["net_profit_eur"])) for v in verdicts]
        assert sort_keys == sorted(sort_keys, reverse=True)

        by_sku = {v["sku"]: v for v in verdicts}
        for item in req.items:
            v = by_sku[item.sku]
            fees = item.target_resale_eur * (fee_pct + reserve_pct)
            net_profit = round(item.target_resale_eur - (item.supplier_price_eur + item.shipping_cost_eur + fees), 2)
            net_margin = round((item.target_resale_eur - (item.supplier_price_eur + item.shipping_cost_eur + fees)) / item.target_resale_eur, 4)
            stage1_fail = (not item.in_stock) or (not item.ships_from_eu) or (item.delivery_days > 5) or (item.rating < 4.2)
            stage2_fail = (net_profit < min_profit) or (net_margin < min_margin)
            if stage1_fail:
                assert v["approved"] is False
                assert v["score"] == 0.0
            elif stage2_fail:
                assert v["approved"] is False
                assert v["score"] == round(max(0.0, net_margin * 100.0), 1)
            else:
                expected_score = round(
                    min(100.0, min(55.0, net_margin * 140.0) + (item.rating / 5.0) * 25.0 + max(0.0, (6 - item.delivery_days) * 4.0)),
                    1,
                )
                assert v["approved"] is True
                assert v["score"] == expected_score


def test_fuzz_lead_budget_optout_spam_and_panel_solar_parity(tmp_path: Path) -> None:
    rng = random.Random(20261125)
    assert classify_lead_service("Quiero un panel solar de 500W para mi casa") is None
    assert classify_lead_service("Necesito automatizar un panel de control con FastAPI") == "b2b_automation"

    fixed_cases = [
        ("Presupuesto de $3k para empezar", "HANDOFF_READY", 3000.0),
        ("Tenemos $500k de fondos", "NEEDS_BUDGET", None),
        ("Con €500k de inversion", "NEEDS_BUDGET", None),
        ("Budget is $500M total", "NEEDS_BUDGET", None),
        ("Presupuesto 500k EUR", "NEEDS_BUDGET", None),
        ("Pago €120.3456 exactos", "NEEDS_BUDGET", None),
        ("Importe 100.5000 EUR", "NEEDS_BUDGET", None),
        ("Entre $10 y $1,500 disponibles", "HANDOFF_READY", 1500.0),
        ("Presupuesto $10 o $1,500", "HANDOFF_READY", 1500.0),
        ("No me escribas mas, darme de baja RGPD — presupuesto 1.200€", "COMPLAINT_OPTOUT", None),
        ("Stop messaging GDPR budget $1,500", "COMPLAINT_OPTOUT", None),
        ("Please opt out and delete my data 900 EUR", "COMPLAINT_OPTOUT", None),
        ("Click here to claim your bonus now!", "SPAM_BLOCKED", None),
        ("Necesito presupuesto para panel solar de 500€", "HANDOFF_READY", 500.0),
    ]

    with _build_client(tmp_path, rpm=500) as client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        for idx, (text, exp_status, exp_budget) in enumerate(fixed_cases):
            resp = client.post(
                "/v1/leads/webhook",
                headers=headers,
                json={"channel": "web", "message_id": f"fixed-lead-{idx}", "sender_id": "s1", "text": text},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == exp_status
            assert body["estimated_budget_eur"] == exp_budget

        for i in range(150):
            mode = i % 3
            if mode == 0:
                amt = rng.randint(50, 250_000)
                text = rng.choice([f"Presupuesto de {amt}€", f"Budget ${amt}", f"Tenemos {amt} EUR"])
                exp_status, exp_budget = "HANDOFF_READY", float(amt)
            elif mode == 1:
                bad_amt = rng.choice(["$500k", "€500k", "$2M", "4k video", "3kg EUR", "€120.3456", "100.5000 EUR", "$1000000"])
                text = f"Consulta comercial sobre {bad_amt}"
                exp_status, exp_budget = "NEEDS_BUDGET", None
            else:
                opt = rng.choice(["GDPR delete my data", "baja RGPD", "suprimir mis datos", "darme de baja", "unsubscribe", "do not contact"])
                text = f"Por favor {opt}, presupuesto {rng.randint(100, 5000)}€"
                exp_status, exp_budget = "COMPLAINT_OPTOUT", None

            resp = client.post(
                "/v1/leads/webhook",
                headers=headers,
                json={"channel": "whatsapp", "message_id": f"fuzz-lead-{i}", "sender_id": f"u-{i}", "text": text},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == exp_status
            assert body["estimated_budget_eur"] == exp_budget


def test_auth_ban_non_bypass_audit_resilience_and_concurrent_burst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Auth-ban non-bypass: exhausting auth_fail_limiter blocks subsequent VALID keys from same IP
    with _build_client(tmp_path / "ban", rpm=5) as ban_client:
        for i in range(10):
            r = ban_client.get("/v1/audit/recent", headers={"X-API-Key": f"bad-{i}"})
            assert r.status_code == 401
        bypass_attempt = ban_client.get("/v1/audit/recent", headers={"X-API-Key": "kl-test-secret-key"})
        assert bypass_attempt.status_code == 429
        assert "Retry-After" in bypass_attempt.headers

    # 2. Audit middleware fault-injection: SQLite failure in store.log_request must NOT crash /health
    with _build_client(tmp_path / "audit_err") as audit_client:
        monkeypatch.setattr(
            audit_client.app.state.store,  # type: ignore[attr-defined]
            "log_request",
            lambda *a, **kw: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
        )
        health_resp = audit_client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"
        assert "X-Request-ID" in health_resp.headers

    # 3. 24-thread concurrent burst (120 mixed requests) with connection leak verification
    opened_conns: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened_conns.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    settings = GatewaySettings(
        api_keys=("kl-test-secret-key",),
        db_path=tmp_path / "burst" / "gateway.db",
        rate_limit_requests=500,
        rate_limit_window_sec=60,
    )
    with TestClient(create_app(settings)) as burst_client:
        headers = {"X-API-Key": "kl-test-secret-key"}
        seed = burst_client.post(
            "/v1/rag/documents",
            headers=headers,
            json={
                "doc_id": "BURST-DOC",
                "title": "Burst SLA",
                "content": "Article 1: Uptime\nConcurrent SQLite WAL pooling handles burst traffic cleanly.",
            },
        )
        assert seed.status_code == 200

        def run_op(idx: int) -> tuple[str, int, dict[str, object]]:
            kind = idx % 5
            if kind == 0:
                r = burst_client.get("/health")
                return ("health", r.status_code, {})
            if kind == 1:
                r = burst_client.post(
                    "/v1/sieve/evaluate",
                    headers=headers,
                    json={
                        "items": [
                            {
                                "sku": f"B-{idx}",
                                "title": "Burst SKU",
                                "supplier_url": "https://supplier.example.com/b",
                                "supplier_price_eur": 30.0,
                                "target_resale_eur": 80.0,
                            }
                        ]
                    },
                )
                return ("sieve", r.status_code, {})
            if kind == 2:
                r = burst_client.post(
                    "/v1/leads/webhook",
                    headers=headers,
                    json={
                        "channel": "web",
                        "message_id": "burst-shared-wamid",
                        "sender_id": "burst-lead",
                        "text": "Presupuesto 2.000 EUR",
                    },
                )
                return ("webhook", r.status_code, r.json())
            if kind == 3:
                r = burst_client.post(
                    "/v1/rag/query",
                    headers=headers,
                    json={"query": "concurrent sqlite wal pooling"},
                )
                return ("rag", r.status_code, {})
            r = burst_client.get("/v1/audit/recent?limit=10", headers=headers)
            return ("audit", r.status_code, {})

        with ThreadPoolExecutor(max_workers=24) as pool:
            outcomes = list(pool.map(run_op, range(120)))

        assert all(code == 200 for _, code, _ in outcomes)
        webhook_bodies = [body for kind, _, body in outcomes if kind == "webhook"]
        assert len(webhook_bodies) == 24
        assert sum(1 for b in webhook_bodies if b["duplicate"] is False) == 1

    assert len(opened_conns) > 0
    for conn in opened_conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


@pytest.mark.parametrize(
    ("text", "expected_budget"),
    [
        ("$3k", 3000.0),
        ("€1.5k", 1500.0),
        ("2,5k€", 2500.0),
        ("10k EUR", 10000.0),
        ("$0.2M", 200000.0),
        ("0.15M EUR", 150000.0),
        ("$0.05k", 50.0),
        ("$0.050k", 50.0),
        ("€0.125k", 125.0),
        ("$0.001M", 1000.0),
        ("$0.049k", None),
        ("€0,049k", None),
        ("$0.0495k", None),
        ("€120.3456", None),
        ("1.200,50€", 1200.5),
        ("$250k", 250000.0),
        ("RGPD audit & ISO27001 $1.5k", 1500.0),
        ("HTTP 200 $1.5k", 1500.0),
        ("order #1234 $500", 500.0),
        ("v2 $3k", 3000.0),
        ("1500€ 2 semanas", 1500.0),
        ("500 europe", None),
        ("$10 setup, total $1,500", 1500.0),
        ("€5/mes o $3k anual", 3000.0),
        ("$500k o $2k", 2000.0),
        ("$49.99 y luego $50", 50.0),
        ("Desde 25€ hasta 1.200€", 1200.0),
        ("4k video", None),
        ("100k followers", None),
        ("3kg EUR", None),
        ("€3kg", None),
        ("$500k", None),
        ("$1M", None),
        ("$250.01k", None),
        ("$3k5", None),
    ],
)
def test_extract_budget_eur_km_scaling_and_multi_amount_alignment(
    tmp_path: Path, text: str, expected_budget: float | None
) -> None:
    assert _extract_budget_eur(text) == expected_budget
    with _build_client(tmp_path) as client:
        resp = client.post(
            "/v1/leads/webhook",
            headers={"X-API-Key": "kl-test-secret-key"},
            json={
                "channel": "web",
                "message_id": f"km-{abs(hash(text))}",
                "sender_id": "lead-km",
                "text": f"Proyecto: {text}",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        expected_status = "HANDOFF_READY" if expected_budget is not None else "NEEDS_BUDGET"
        assert body["status"] == expected_status
        assert body["estimated_budget_eur"] == expected_budget


@pytest.mark.parametrize(
    ("text", "expected_status", "expected_budget"),
    [
        ("Are you GDPR compliant?", "NEEDS_BUDGET", None),
        ("¿Vuestro bot cumple con RGPD?", "NEEDS_BUDGET", None),
        ("We need a GDPR RAG API budget $1500", "HANDOFF_READY", 1500.0),
        ("Necesitamos bot RGPD presupuesto 900 EUR", "HANDOFF_READY", 900.0),
        ("Necesito API workflow con DPA GDPR presupuesto 2.000€", "HANDOFF_READY", 2000.0),
        ("suprimir mis datos", "COMPLAINT_OPTOUT", None),
        ("Quiero supresión de mis datos bajo RGPD", "COMPLAINT_OPTOUT", None),
        ("Solicito supresión de datos", "COMPLAINT_OPTOUT", None),
        ("GDPR delete my data", "COMPLAINT_OPTOUT", None),
        ("GDPR removal", "COMPLAINT_OPTOUT", None),
        ("GDPR deletion", "COMPLAINT_OPTOUT", None),
        ("Quiero darme de baja por RGPD", "COMPLAINT_OPTOUT", None),
        ("Solicito baja RGPD", "COMPLAINT_OPTOUT", None),
        ("Quiero borrar todos mis datos bajo RGPD presupuesto 1.000€", "COMPLAINT_OPTOUT", None),
        ("Suprima nuestros datos RGPD", "COMPLAINT_OPTOUT", None),
        ("GDPR opt-out", "COMPLAINT_OPTOUT", None),
        ("RGPD erasure", "COMPLAINT_OPTOUT", None),
        ("reclamación RGPD", "COMPLAINT_OPTOUT", None),
        ("denuncia AEPD RGPD", "COMPLAINT_OPTOUT", None),
        ("report you to ICO GDPR", "COMPLAINT_OPTOUT", None),
    ],
)
def test_lead_webhook_gdpr_rgpd_context_alignment(
    tmp_path: Path, text: str, expected_status: str, expected_budget: float | None
) -> None:
    with _build_client(tmp_path) as client:
        resp = client.post(
            "/v1/leads/webhook",
            headers={"X-API-Key": "kl-test-secret-key"},
            json={
                "channel": "whatsapp",
                "message_id": f"gdpr-{abs(hash(text))}",
                "sender_id": "lead-gdpr",
                "text": text,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == expected_status
        assert body["estimated_budget_eur"] == expected_budget


class _GatewaySpyConn:
    def __init__(self, inner: sqlite3.Connection, sql_log: list[str], fail_hook: Any = None) -> None:
        self._inner = inner
        self._sql_log = sql_log
        self._fail_hook = fail_hook
        self.closed = False

    @property
    def row_factory(self) -> Any:
        return self._inner.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._inner.row_factory = value

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        normalized = " ".join(sql.split()).rstrip(";").upper()
        self._sql_log.append(normalized)
        if self._fail_hook is not None:
            self._fail_hook("execute", normalized)
        return self._inner.execute(sql, *args, **kwargs)

    def executescript(self, sql_script: str) -> sqlite3.Cursor:
        self._sql_log.append("EXECUTESCRIPT")
        if self._fail_hook is not None:
            self._fail_hook("executescript", sql_script)
        return self._inner.executescript(sql_script)

    def commit(self) -> None:
        self._inner.commit()

    def close(self) -> None:
        self.closed = True
        self._inner.close()

    def __enter__(self) -> _GatewaySpyConn:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> bool | None:
        return self._inner.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_gateway_store_pragma_order_and_skips_redundant_wal(monkeypatch: pytest.MonkeyPatch) -> None:
    sql_log: list[str] = []
    timeouts: list[object] = []
    real_connect = sqlite3.connect

    def spy_connect(*args: object, **kwargs: object) -> _GatewaySpyConn:
        timeouts.append(kwargs.get("timeout"))
        return _GatewaySpyConn(real_connect(*args, **kwargs), sql_log)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "gateway_pragma.db"
        with GatewayStore(db_path) as store:
            assert timeouts == [5.0]
            busy_idx = sql_log.index("PRAGMA BUSY_TIMEOUT=5000")
            mode_idx = sql_log.index("PRAGMA JOURNAL_MODE")
            wal_idx = sql_log.index("PRAGMA JOURNAL_MODE=WAL")
            assert busy_idx < mode_idx < wal_idx

        sql_log.clear()
        with GatewayStore(db_path) as store:
            assert "PRAGMA BUSY_TIMEOUT=5000" in sql_log
            assert "PRAGMA JOURNAL_MODE" in sql_log
            assert "PRAGMA JOURNAL_MODE=WAL" not in sql_log

            sql_log.clear()
            err_holder: list[BaseException] = []

            def worker() -> None:
                try:
                    assert store.record_webhook_once("web", "w-1", "s-1", "NEEDS_BUDGET") is True
                except BaseException as exc:  # pragma: no cover
                    err_holder.append(exc)

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(worker).result()
            assert err_holder == []
            assert "PRAGMA BUSY_TIMEOUT=5000" in sql_log
            assert "PRAGMA JOURNAL_MODE=WAL" not in sql_log

    with GatewayStore(":memory:") as mem_store:
        assert mem_store.journal_mode() == "memory"


def test_gateway_store_cold_start_retry_and_tempdir_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = sqlite3.connect
    monkeypatch.setattr(time, "sleep", lambda _secs: None)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "retry.db"
        conns: list[_GatewaySpyConn] = []
        sql_log: list[str] = []

        def flaky_wal_connect(*args: object, **kwargs: object) -> _GatewaySpyConn:
            attempt = len(conns) + 1

            def fail_hook(kind: str, sql: str) -> None:
                if kind == "execute" and sql == "PRAGMA JOURNAL_MODE=WAL" and attempt <= 2:
                    raise sqlite3.OperationalError("database is locked")

            spy = _GatewaySpyConn(real_connect(*args, **kwargs), sql_log, fail_hook)  # type: ignore[arg-type]
            conns.append(spy)
            return spy

        monkeypatch.setattr(sqlite3, "connect", flaky_wal_connect)
        store = GatewayStore(db_path)
        assert len(conns) == 3
        assert conns[0].closed and conns[1].closed
        assert not conns[2].closed
        assert store.record_webhook_once("whatsapp", "m-retry", "s-retry", "HANDOFF_READY") is True
        store.close()
        assert conns[2].closed

    assert not tmp_dir.exists()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "always_locked_schema.db"
        conns = []
        sql_log = []

        def schema_locked_connect(*args: object, **kwargs: object) -> _GatewaySpyConn:
            def fail_hook(kind: str, _sql: str) -> None:
                if kind == "executescript":
                    raise sqlite3.OperationalError("database is locked")

            spy = _GatewaySpyConn(real_connect(*args, **kwargs), sql_log, fail_hook)  # type: ignore[arg-type]
            conns.append(spy)
            return spy

        monkeypatch.setattr(sqlite3, "connect", schema_locked_connect)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            GatewayStore(db_path)
        assert len(conns) == 6
        assert all(c.closed for c in conns)

    assert not tmp_dir.exists()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "non_lock.db"
        conns = []
        sql_log = []

        def non_lock_connect(*args: object, **kwargs: object) -> _GatewaySpyConn:
            def fail_hook(kind: str, sql: str) -> None:
                if kind == "execute" and sql == "PRAGMA JOURNAL_MODE=WAL":
                    raise sqlite3.OperationalError("disk I/O error")

            spy = _GatewaySpyConn(real_connect(*args, **kwargs), sql_log, fail_hook)  # type: ignore[arg-type]
            conns.append(spy)
            return spy

        monkeypatch.setattr(sqlite3, "connect", non_lock_connect)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            GatewayStore(db_path)
        assert len(conns) == 1
        assert conns[0].closed

    assert not tmp_dir.exists()
