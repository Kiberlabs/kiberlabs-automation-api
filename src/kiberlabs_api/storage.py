"""Concurrent SQLite WAL + FTS5 storage backing API audit logs, webhook idempotency, and RAG search."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS webhook_events (
    channel TEXT NOT NULL,
    message_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (channel, message_id)
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    section_ref TEXT NOT NULL,
    text TEXT NOT NULL,
    chunk_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks (doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    section_ref,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class GatewayStore:
    """Thread-safe SQLite WAL + FTS5 persistence layer shared by all gateway endpoints."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize SQLite in WAL mode (busy_timeout=5000ms) and bootstrap schema tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        """Open a per-operation SQLite connection configured for concurrent WAL readers/writers."""
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def journal_mode(self) -> str:
        """Return active SQLite journal mode (expected: 'wal')."""
        with self._connect() as conn:
            row = conn.execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row else ""

    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Persist one HTTP request telemetry row to api_audit_log."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_audit_log (request_id, method, path, status_code, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (request_id, method, path, status_code, round(latency_ms, 2), now_iso),
            )

    def recent_audit(self, limit: int = 20) -> list[dict[str, object]]:
        """Return the latest API audit log records ordered newest-first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT request_id, method, path, status_code, latency_ms, created_at "
                "FROM api_audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_webhook_once(
        self,
        channel: str,
        message_id: str,
        sender_id: str,
        status: str,
    ) -> bool:
        """Atomically insert (channel, message_id) into webhook_events.

        Returns True on first delivery, or False if the webhook is a duplicate retry.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO webhook_events (channel, message_id, sender_id, status, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (channel, message_id, sender_id, status, now_iso),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def upsert_rag_chunks(self, doc_id: str, rows: list[tuple[str, str, str, str, str]]) -> int:
        """Atomically replace all indexed chunks for doc_id across both rag_chunks and FTS5."""
        with self._connect() as conn:
            old_ids = [
                r["chunk_id"]
                for r in conn.execute(
                    "SELECT chunk_id FROM rag_chunks WHERE doc_id = ?", (doc_id,)
                ).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                conn.execute(
                    f"DELETE FROM rag_chunks_fts WHERE chunk_id IN ({placeholders})", old_ids
                )
            conn.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))

            for chunk_id, title, section_ref, text, chunk_hash in rows:
                conn.execute(
                    """
                    INSERT INTO rag_chunks (chunk_id, doc_id, title, section_ref, text, chunk_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, title, section_ref, text, chunk_hash),
                )
                conn.execute(
                    "INSERT INTO rag_chunks_fts (chunk_id, title, section_ref, text) VALUES (?, ?, ?, ?)",
                    (chunk_id, title, section_ref, text),
                )
        return len(rows)

    def search_rag_fts(self, fts_expr: str, top_k: int = 3) -> list[dict[str, object]]:
        """Execute a BM25-ranked FTS5 query and return up to top_k grounded chunks."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    c.title,
                    c.section_ref,
                    c.text,
                    c.chunk_hash,
                    bm25(rag_chunks_fts, 1.0, 2.0, 1.0) AS raw_bm25
                FROM rag_chunks_fts
                JOIN rag_chunks c ON c.chunk_id = rag_chunks_fts.chunk_id
                WHERE rag_chunks_fts MATCH ?
                ORDER BY raw_bm25 ASC
                LIMIT ?
                """,
                (fts_expr, top_k),
            ).fetchall()
            return [dict(r) for r in rows]
