"""Domain service layer exposing KiberLabs deterministic engines behind typed Pydantic models."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kiberlabs_api.security import validate_public_http_url
from kiberlabs_api.storage import GatewayStore

_MAX_SAFE_AMOUNT = 1e12
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")

_HEADING_RE = re.compile(
    r"^(?:#{1,3}\s+|(?:Article|Art[ií]culo|Section|Secci[oó]n|Clause)\s+[0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)
_AMT = r"([1-9]\d{0,2}(?:[.,]\d{3})+|\d{1,6})([.,]\d{1,3}(?=[kKmM])|[.,]\d{1,2})?([kKmM])?(?!\d|[.,]\d)"
_BUDGET_RE = re.compile(
    rf"(?:(?<![A-Za-zÀ-ÿ\d.,])[€$]\s*{_AMT}(?![A-Za-zÀ-ÿ])|(?<![A-Za-zÀ-ÿ\d.,]){_AMT}\s*(?:(?:€|\$)(?![A-Za-zÀ-ÿ\d])|(?:eur|euros?|usd)\b(?![A-Za-zÀ-ÿ])))",
    re.IGNORECASE,
)


def _clean_non_empty(value: str, min_len: int, field_name: str) -> str:
    cleaned = "".join(
        ch for ch in _ZERO_WIDTH_RE.sub("", value.replace("\x00", "")) if unicodedata.category(ch) != "Cf"
    ).strip()
    if len(cleaned) < min_len:
        raise ValueError(f"{field_name} must contain at least {min_len} non-whitespace characters")
    return cleaned


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric inputs")
    return value


def _extract_budget_eur(text: str) -> float | None:
    """Parse first European/US currency expression (with optional k/M) within 50–250,000 EUR."""
    norm_text = unicodedata.normalize("NFKC", _ZERO_WIDTH_RE.sub("", text))
    for match in _BUDGET_RE.finditer(norm_text):
        int_part = match.group(1) or match.group(4) or ""
        dec_part = match.group(2) or match.group(5) or ""
        suffix = (match.group(3) or match.group(6) or "").lower()
        raw = int_part.replace(".", "").replace(",", "")
        if dec_part:
            raw = f"{raw}.{dec_part[1:]}"
        value = float(raw)
        if suffix == "k":
            value *= 1_000.0
        elif suffix == "m":
            value *= 1_000_000.0
        if 50.0 <= value <= 250_000.0:
            return value
    return None


def _compile_word_matcher(phrases: tuple[str, ...]) -> re.Pattern[str]:
    alts = "|".join(r"\s+".join(map(re.escape, p.split())) for p in phrases)
    return re.compile(rf"\b(?:{alts})\b")


def _normalize_lead_text(text: str) -> str:
    folded = _fold_ascii(text)
    return folded.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": "'",
                "\u201b": "'",
                "\u2032": "'",
                "\u02bc": "'",
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
            }
        )
    )


_SPAM_PHRASES = (
    "crypto airdrop",
    "casino bonus",
    "forex signal",
    "buy followers",
    "click here to claim",
    "gana dinero rapido",
)
_COMPLAINT_OPTOUT_PATTERNS = (
    "stop messaging",
    "stop contacting",
    "do not contact",
    "dont contact",
    "don't contact",
    "unsubscribe",
    "opt out",
    "opt-out",
    "remove my",
    "delete my data",
    "erase my data",
    "remove my data",
    "no me escribas",
    "no me contactes",
    "deja de escribir",
    "darme de baja",
    "darse de baja",
    "dar de baja",
    "quita mi anuncio",
    "borra mis datos",
    "borrar mis datos",
    "elimina mis datos",
    "eliminar mis datos",
    "suprimir mis datos",
    "supresion de mis datos",
    "supresion de datos",
    "derecho al olvido",
    "denunciar",
)
_SERVICE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "web_scraping": (
        "scraping",
        "scraper",
        "monitor de precios",
        "monitorear precios",
        "seguimiento de precios",
        "extraer precios",
        "catalogo",
        "price monitor",
        "extraer datos",
    ),
    "b2b_automation": (
        "automatizacion",
        "automatizar",
        "automation",
        "workflow",
        "api",
        "fastapi",
        "docker",
        "rag",
        "dashboard",
        "panel de control",
        "panel admin",
        "panel administrativo",
        "panel web",
        "panel b2b",
        "admin panel",
    ),
    "messaging_bot": ("bot", "whatsapp", "telegram", "chatbot", "atencion automatica"),
    "ecommerce_ai": ("tienda", "ecommerce", "shopify", "woocommerce", "video", "ffmpeg"),
}
_SPAM_RE = _compile_word_matcher(_SPAM_PHRASES)
_OPTOUT_RE = _compile_word_matcher(_COMPLAINT_OPTOUT_PATTERNS)
_GDPR_OPTOUT_ACTION = (
    r"(?:(?:delete|erase|remove|borra\w*|elimina\w*|suprim\w*)\s+"
    r"(?:all\s+|todos\s+)?(?:my|our|mis|nuestros)\b"
    r"|baja|opt[- ]?out|erasure|complaint|reclamacion|denuncia\w*"
    r"|report\s+(?:you|this|it|to)\b|aepd|ico)"
)
_GDPR_OPTOUT_RE = re.compile(
    rf"\b(?:(?:gdpr|rgpd)\b.{{0,30}}\b{_GDPR_OPTOUT_ACTION}"
    rf"|{_GDPR_OPTOUT_ACTION}.{{0,30}}\b(?:gdpr|rgpd)\b"
    rf"|(?:gdpr|rgpd)\s+(?:removal|deletion)\b)"
)
_SERVICE_MATCHERS = tuple((cat, _compile_word_matcher(kws)) for cat, kws in _SERVICE_KEYWORDS.items())

_STOPWORDS = frozenset(
    {
        "to", "in", "of", "on", "at", "by", "an", "is", "as",
        "de", "en", "el", "la", "es", "un", "al",
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "what", "when", "how",
        "does", "under", "about", "into", "your", "have", "must", "can", "will",
        "los", "las", "del", "una", "uno", "para", "con", "que", "como", "cual", "cuando",
        "sobre", "por", "son", "esta", "este", "puede", "debe", "entre",
    }
)


class SieveSKURequest(BaseModel):
    """Inbound SKU record for deterministic net-margin evaluation."""

    model_config = ConfigDict(allow_inf_nan=False)

    sku: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    supplier_url: str
    supplier_price_eur: float = Field(gt=0, le=_MAX_SAFE_AMOUNT)
    target_resale_eur: float = Field(gt=0, le=_MAX_SAFE_AMOUNT)
    shipping_cost_eur: float = Field(ge=0, le=_MAX_SAFE_AMOUNT, default=0.0)
    in_stock: bool = True
    ships_from_eu: bool = True
    delivery_days: int = Field(ge=0, le=90, default=3)
    rating: float = Field(ge=0.0, le=5.0, default=4.5)

    @field_validator("supplier_price_eur", "target_resale_eur", "shipping_cost_eur", "delivery_days", "rating", mode="before")
    @classmethod
    def _reject_bool_numbers(cls, value: object) -> object:
        return _reject_bool(value)

    @field_validator("sku", "title")
    @classmethod
    def _clean_text_fields(cls, value: str, info: object) -> str:
        return _clean_non_empty(value, 1, getattr(info, "field_name", "field"))

    @field_validator("supplier_url")
    @classmethod
    def _ssrf_guard(cls, value: str) -> str:
        """Enforce SSRF boundary before supplier URL enters downstream pipelines."""
        return validate_public_http_url(value)


class SieveBatchRequest(BaseModel):
    """Batch request for /v1/sieve/evaluate."""

    model_config = ConfigDict(allow_inf_nan=False)

    items: list[SieveSKURequest] = Field(min_length=1, max_length=200)
    marketplace_fee_pct: float = Field(ge=0.0, lt=0.8, default=0.12)
    return_reserve_pct: float = Field(ge=0.0, lt=0.5, default=0.04)
    min_net_margin_pct: float = Field(ge=0.0, lt=1.0, default=0.18)
    min_net_profit_eur: float = Field(ge=0.0, le=_MAX_SAFE_AMOUNT, default=8.0)

    @field_validator("marketplace_fee_pct", "return_reserve_pct", "min_net_margin_pct", "min_net_profit_eur", mode="before")
    @classmethod
    def _reject_bool_numbers(cls, value: object) -> object:
        return _reject_bool(value)

    @model_validator(mode="after")
    def _validate_economics_bounds(self) -> SieveBatchRequest:
        if self.marketplace_fee_pct + self.return_reserve_pct >= 1.0:
            raise ValueError("marketplace_fee_pct + return_reserve_pct must be < 1.0")
        for item in self.items:
            fees = item.target_resale_eur * (self.marketplace_fee_pct + self.return_reserve_pct)
            raw_profit = item.target_resale_eur - (item.supplier_price_eur + item.shipping_cost_eur + fees)
            raw_margin = raw_profit / item.target_resale_eur
            if not (math.isfinite(raw_profit) and math.isfinite(raw_margin)):
                raise ValueError("Computed economic metrics overflowed to non-finite values")
        return self


class LeadWebhookRequest(BaseModel):
    """Normalized inbound WhatsApp/Telegram/Web webhook payload."""

    channel: Literal["whatsapp", "telegram", "web"]
    message_id: str = Field(min_length=2, max_length=120)
    sender_id: str = Field(min_length=2, max_length=80)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("message_id", "sender_id")
    @classmethod
    def _clean_ids(cls, value: str, info: object) -> str:
        return _clean_non_empty(value, 2, getattr(info, "field_name", "field"))

    @field_validator("text")
    @classmethod
    def _clean_body(cls, value: str) -> str:
        return _clean_non_empty(value, 1, "text")


class RagDocumentRequest(BaseModel):
    """Structured regulatory or policy document submitted to /v1/rag/documents."""

    doc_id: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=3, max_length=200)
    content: str = Field(min_length=20, max_length=200_000)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        return _clean_non_empty(value, 3, "title")

    @field_validator("content")
    @classmethod
    def _clean_content(cls, value: str) -> str:
        return _clean_non_empty(value, 20, "content")


class RagQueryRequest(BaseModel):
    """Natural-language retrieval request for /v1/rag/query."""

    model_config = ConfigDict(allow_inf_nan=False)

    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    min_token_overlap: float = Field(default=0.25, ge=0.05, le=1.0)

    @field_validator("top_k", "min_token_overlap", mode="before")
    @classmethod
    def _reject_bool_numbers(cls, value: object) -> object:
        return _reject_bool(value)

    @field_validator("query")
    @classmethod
    def _clean_query(cls, value: str) -> str:
        return _clean_non_empty(value, 2, "query")


def evaluate_sieve_batch(req: SieveBatchRequest) -> dict[str, object]:
    """Run deterministic 3-stage net-margin filtering and return ranked approval verdicts."""
    verdicts: list[dict[str, object]] = []
    for item in req.items:
        fees = item.target_resale_eur * (req.marketplace_fee_pct + req.return_reserve_pct)
        raw_profit = item.target_resale_eur - (item.supplier_price_eur + item.shipping_cost_eur + fees)
        raw_margin = raw_profit / item.target_resale_eur
        if not (math.isfinite(raw_profit) and math.isfinite(raw_margin)):
            raise ValueError("Computed economic metrics overflowed to non-finite values")
        net_profit = round(raw_profit, 2)
        net_margin = round(raw_margin, 4)

        hard_reasons: list[str] = []
        if not item.in_stock:
            hard_reasons.append("out_of_stock")
        if not item.ships_from_eu:
            hard_reasons.append("non_eu_dispatch")
        if item.delivery_days > 5:
            hard_reasons.append("delivery_days>5")
        if item.rating < 4.2:
            hard_reasons.append("rating<4.2")

        if hard_reasons:
            approved = False
            reasons = hard_reasons
            score = 0.0
        else:
            econ_reasons: list[str] = []
            if net_profit < req.min_net_profit_eur:
                econ_reasons.append(f"net_profit<{req.min_net_profit_eur:.2f}")
            if net_margin < req.min_net_margin_pct:
                econ_reasons.append(f"net_margin<{req.min_net_margin_pct:.0%}")

            if econ_reasons:
                approved = False
                reasons = econ_reasons
                score = round(max(0.0, net_margin * 100.0), 1)
            else:
                approved = True
                reasons = ["passed_all_gates"]
                score = round(
                    min(
                        100.0,
                        min(55.0, net_margin * 140.0)
                        + (item.rating / 5.0) * 25.0
                        + max(0.0, (6 - item.delivery_days) * 4.0),
                    ),
                    1,
                )

        verdicts.append(
            {
                "sku": item.sku,
                "title": item.title,
                "approved": approved,
                "net_profit_eur": net_profit,
                "net_margin_pct": net_margin,
                "score": score,
                "reasons": reasons,
            }
        )

    verdicts.sort(key=lambda v: (bool(v["approved"]), float(v["score"]), float(v["net_profit_eur"])), reverse=True)
    approved_count = sum(1 for v in verdicts if v["approved"])
    return {
        "processed": len(verdicts),
        "approved_count": approved_count,
        "verdicts": verdicts,
    }


def classify_lead_service(text: str) -> str | None:
    """Classify lead service category using word-boundary keywords without bare 'panel' false positives."""
    norm = _normalize_lead_text(text)
    for category, pattern in _SERVICE_MATCHERS:
        if pattern.search(norm) is not None:
            return category
    return None


def handle_lead_webhook(req: LeadWebhookRequest, store: GatewayStore) -> dict[str, object]:
    """Process a 24/7 lead webhook with atomic SQLite WAL deduplication and zero-token triage."""
    norm = _normalize_lead_text(req.text)
    is_spam = norm.count("http://") + norm.count("https://") >= 3 or _SPAM_RE.search(norm) is not None
    is_optout = _OPTOUT_RE.search(norm) is not None or _GDPR_OPTOUT_RE.search(norm) is not None

    budget_eur: float | None = None
    if is_spam:
        status = "SPAM_BLOCKED"
    elif is_optout:
        status = "COMPLAINT_OPTOUT"
    else:
        budget_eur = _extract_budget_eur(req.text)
        status = "HANDOFF_READY" if budget_eur is not None else "NEEDS_BUDGET"

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
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    nfkd = unicodedata.normalize("NFKD", cleaned)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


def _split_section_text(text: str, max_chars: int) -> list[str]:
    """Split oversized section bodies on whitespace into bounded chunks <= max_chars."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in cleaned.split(" "):
        if len(word) > max_chars:
            if buf:
                parts.append(" ".join(buf))
                buf, buf_len = [], 0
            for start in range(0, len(word), max_chars):
                parts.append(word[start : start + max_chars])
            continue
        if buf and buf_len + 1 + len(word) > max_chars:
            parts.append(" ".join(buf))
            buf = [word]
            buf_len = len(word)
        else:
            buf.append(word)
            buf_len += len(word) + (1 if buf_len else 0)
    if buf:
        parts.append(" ".join(buf))
    return parts


def index_rag_document(req: RagDocumentRequest, store: GatewayStore, max_chars: int = 1200) -> dict[str, object]:
    """Split a document by Article/Section headings and atomically index chunks into SQLite FTS5."""
    if isinstance(max_chars, bool) or max_chars < 50:
        raise ValueError("max_chars must be >= 50")
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
    seq = 0
    for ref, lines in sections:
        body = " ".join(lines).strip()
        for part_idx, segment in enumerate(_split_section_text(body, max_chars), start=1):
            seq += 1
            part_ref = ref if part_idx == 1 else f"{ref} (part {part_idx})"
            chunk_id = f"{req.doc_id}::c{seq:03d}"
            chunk_hash = hashlib.sha256(f"{req.doc_id}|{part_ref}|{segment}".encode("utf-8")).hexdigest()[:16]
            rows.append((chunk_id, req.title, part_ref, segment, chunk_hash))

    if not rows:
        raise ValueError("Document produced no indexable chunks")
    written = store.upsert_rag_chunks(req.doc_id, rows)
    return {"doc_id": req.doc_id, "chunks_indexed": written}


def query_rag_citations(req: RagQueryRequest, store: GatewayStore) -> dict[str, object]:
    """Sanitize query against FTS5 injection, run BM25 search, and abstain on weak evidence."""
    tokens = list(
        dict.fromkeys(
            t for t in re.findall(r"[a-z0-9]+", _fold_ascii(req.query)) if len(t) >= 2 and t not in _STOPWORDS
        )
    )[:32]
    if not tokens:
        return {"status": "INSUFFICIENT_EVIDENCE", "citations": [], "verified": True}

    fts_expr = " OR ".join(f'"{tok}"' for tok in tokens)
    rows = store.search_rag_fts(fts_expr, top_k=max(req.top_k * 5, 25))

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
    citations = citations[: req.top_k]
    if not citations:
        return {"status": "INSUFFICIENT_EVIDENCE", "citations": [], "verified": True}

    return {"status": "GROUNDED", "citations": citations, "verified": True}
