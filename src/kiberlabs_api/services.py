"""Domain service layer exposing KiberLabs deterministic engines behind typed Pydantic models."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from kiberlabs_api.security import validate_public_http_url
from kiberlabs_api.storage import GatewayStore

_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+|(?:Article|Art[ií]culo|Section|Secci[oó]n|Clause)\s+[0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)
_BUDGET_RE = re.compile(
    r"(?:[€$]\s*(\d{1,3}(?:[.,]\d{3})+|\d{2,6})|(\d{1,3}(?:[.,]\d{3})+|\d{2,6})\s*(?:€|eur|euros?|usd|\$))",
    re.IGNORECASE,
)
_SPAM_PHRASES = ("crypto airdrop", "casino bonus", "forex signal", "buy followers", "gana dinero rapido")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "what", "when", "how",
        "los", "las", "del", "una", "para", "con", "que", "como", "sobre", "por",
    }
)


class SieveSKURequest(BaseModel):
    """Inbound SKU record for deterministic net-margin evaluation."""

    sku: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    supplier_url: str
    supplier_price_eur: float = Field(gt=0)
    target_resale_eur: float = Field(gt=0)
    shipping_cost_eur: float = Field(ge=0, default=0.0)
    in_stock: bool = True
    ships_from_eu: bool = True
    delivery_days: int = Field(ge=0, le=90, default=3)
    rating: float = Field(ge=0.0, le=5.0, default=4.5)

    @field_validator("supplier_url")
    @classmethod
    def _ssrf_guard(cls, value: str) -> str:
        """Enforce SSRF boundary before supplier URL enters downstream pipelines."""
        return validate_public_http_url(value)


class SieveBatchRequest(BaseModel):
    """Batch request for /v1/sieve/evaluate."""

    items: list[SieveSKURequest] = Field(min_length=1, max_length=200)
    marketplace_fee_pct: float = Field(ge=0.0, lt=0.8, default=0.12)
    return_reserve_pct: float = Field(ge=0.0, lt=0.5, default=0.04)
    min_net_margin_pct: float = Field(ge=0.0, lt=1.0, default=0.18)
    min_net_profit_eur: float = Field(ge=0.0, default=8.0)


class LeadWebhookRequest(BaseModel):
    """Normalized inbound WhatsApp/Telegram/Web webhook payload."""

    channel: Literal["whatsapp", "telegram", "web"]
    message_id: str = Field(min_length=2, max_length=120)
    sender_id: str = Field(min_length=2, max_length=80)
    text: str = Field(min_length=1, max_length=2000)


class RagDocumentRequest(BaseModel):
    """Structured regulatory or policy document submitted to /v1/rag/documents."""

    doc_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=3, max_length=200)
    content: str = Field(min_length=20)


class RagQueryRequest(BaseModel):
    """Natural-language retrieval request for /v1/rag/query."""

    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    min_token_overlap: float = Field(default=0.25, ge=0.05, le=1.0)


def evaluate_sieve_batch(req: SieveBatchRequest) -> dict[str, object]:
    """Run deterministic 3-stage net-margin filtering and return ranked approval verdicts."""
    verdicts: list[dict[str, object]] = []
    for item in req.items:
        fees = item.target_resale_eur * (req.marketplace_fee_pct + req.return_reserve_pct)
        net_profit = round(item.target_resale_eur - (item.supplier_price_eur + item.shipping_cost_eur + fees), 2)
        net_margin = round(net_profit / item.target_resale_eur, 4)

        reasons: list[str] = []
        if not item.in_stock:
            reasons.append("out_of_stock")
        if not item.ships_from_eu:
            reasons.append("non_eu_dispatch")
        if item.delivery_days > 5:
            reasons.append("delivery_days>5")
        if item.rating < 4.2:
            reasons.append("rating<4.2")
        if net_profit < req.min_net_profit_eur:
            reasons.append(f"net_profit<{req.min_net_profit_eur:.2f}")
        if net_margin < req.min_net_margin_pct:
            reasons.append(f"net_margin<{req.min_net_margin_pct:.0%}")

        approved = len(reasons) == 0
        score = (
            round(min(100.0, min(60.0, net_margin * 150.0) + (item.rating / 5.0) * 25.0 + max(0.0, (6 - item.delivery_days) * 3.0)), 1)
            if approved
            else 0.0
        )
        verdicts.append(
            {
                "sku": item.sku,
                "title": item.title,
                "approved": approved,
                "net_profit_eur": net_profit,
                "net_margin_pct": net_margin,
                "score": score,
                "reasons": reasons or ["passed_all_gates"],
            }
        )

    verdicts.sort(key=lambda v: (bool(v["approved"]), float(v["score"])), reverse=True)
    approved_count = sum(1 for v in verdicts if v["approved"])
    return {
        "processed": len(verdicts),
        "approved_count": approved_count,
        "verdicts": verdicts,
    }


def handle_lead_webhook(req: LeadWebhookRequest, store: GatewayStore) -> dict[str, object]:
    """Process a 24/7 lead webhook with atomic SQLite WAL deduplication and zero-token triage."""
    low = req.text.lower()
    is_spam = low.count("http://") + low.count("https://") >= 3 or any(p in low for p in _SPAM_PHRASES)
    budget_match = _BUDGET_RE.search(req.text)
    budget_eur: float | None = None
    if budget_match:
        raw_num = (budget_match.group(1) or budget_match.group(2) or "").replace(".", "").replace(",", "")
        val = float(raw_num)
        if 50.0 <= val <= 250_000.0:
            budget_eur = val

    if is_spam:
        status = "SPAM_BLOCKED"
    elif budget_eur is not None:
        status = "HANDOFF_READY"
    else:
        status = "NEEDS_BUDGET"

    is_new = store.record_webhook_once(req.channel, req.message_id, req.sender_id, status)
    if not is_new:
        return {
            "duplicate": True,
            "channel": req.channel,
            "message_id": req.message_id,
            "status": "DUPLICATE_IGNORED",
            "estimated_budget_eur": None,
        }

    return {
        "duplicate": False,
        "channel": req.channel,
        "message_id": req.message_id,
        "status": status,
        "estimated_budget_eur": budget_eur,
    }


def _fold_ascii(text: str) -> str:
    """Fold Unicode diacritics into lowercase ASCII for accent-insensitive EN/ES matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


def index_rag_document(req: RagDocumentRequest, store: GatewayStore) -> dict[str, object]:
    """Split a document by Article/Section headings and atomically index chunks into SQLite FTS5."""
    sections: list[tuple[str, list[str]]] = []
    current_ref = "General"
    buf: list[str] = []

    for raw_line in req.content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HEADING_RE.match(line):
            if buf:
                sections.append((current_ref, buf))
                buf = []
            current_ref = line.lstrip("#").strip().rstrip(":")
            buf.append(line.lstrip("#").strip())
        else:
            buf.append(line)
    if buf:
        sections.append((current_ref, buf))

    rows: list[tuple[str, str, str, str, str]] = []
    for idx, (ref, lines) in enumerate(sections, start=1):
        body = " ".join(lines).strip()
        chunk_id = f"{req.doc_id}::c{idx:03d}"
        chunk_hash = hashlib.sha256(f"{req.doc_id}|{ref}|{body}".encode("utf-8")).hexdigest()[:16]
        rows.append((chunk_id, req.title, ref, body, chunk_hash))

    written = store.upsert_rag_chunks(req.doc_id, rows)
    return {"doc_id": req.doc_id, "chunks_indexed": written}


def query_rag_citations(req: RagQueryRequest, store: GatewayStore) -> dict[str, object]:
    """Sanitize query against FTS5 injection, run BM25 search, and abstain on weak evidence."""
    tokens = list(
        dict.fromkeys(
            t for t in re.findall(r"[a-z0-9]+", _fold_ascii(req.query)) if len(t) >= 2 and t not in _STOPWORDS
        )
    )
    if not tokens:
        return {"status": "INSUFFICIENT_EVIDENCE", "citations": [], "verified": True}

    fts_expr = " OR ".join(f'"{tok}"' for tok in tokens)
    rows = store.search_rag_fts(fts_expr, top_k=req.top_k)

    citations: list[dict[str, object]] = []
    for row in rows:
        cand_tokens = set(re.findall(r"[a-z0-9]+", _fold_ascii(f"{row['title']} {row['section_ref']} {row['text']}")))
        overlap = round(sum(1 for t in tokens if t in cand_tokens) / len(tokens), 4)
        if overlap < req.min_token_overlap:
            continue
        citations.append(
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "section_ref": row["section_ref"],
                "token_overlap": overlap,
                "snippet": str(row["text"])[:240],
                "chunk_hash": row["chunk_hash"],
            }
        )

    citations.sort(key=lambda c: float(c["token_overlap"]), reverse=True)
    if not citations:
        return {"status": "INSUFFICIENT_EVIDENCE", "citations": [], "verified": True}

    return {"status": "GROUNDED", "citations": citations, "verified": True}
