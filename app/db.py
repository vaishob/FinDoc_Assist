from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import settings


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source_hash TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    pii_detected INTEGER NOT NULL DEFAULT 0,
                    sensitivity_level TEXT NOT NULL DEFAULT 'standard',
                    summary TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER,
                    section_title TEXT,
                    token_count INTEGER NOT NULL,
                    contains_pii INTEGER NOT NULL DEFAULT 0,
                    pii_types TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS query_logs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    document_ids TEXT NOT NULL,
                    retrieval_latency_ms INTEGER NOT NULL,
                    generation_latency_ms INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS guardrail_events (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_document(
        self,
        document_id: str,
        filename: str,
        stored_path: str,
        content_type: str,
        source_hash: str,
    ) -> None:
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, stored_path, content_type, source_hash, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (document_id, filename, stored_path, content_type, source_hash, now, now),
            )

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        error_message: str | None = None,
        chunk_count: int | None = None,
        pii_detected: bool | None = None,
        summary: str | None = None,
    ) -> None:
        fields: list[str] = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, utcnow()]
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)
        if chunk_count is not None:
            fields.append("chunk_count = ?")
            values.append(chunk_count)
        if pii_detected is not None:
            fields.append("pii_detected = ?")
            values.append(int(pii_detected))
        if summary is not None:
            fields.append("summary = ?")
            values.append(summary)
        values.append(document_id)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(fields)} WHERE id = ?",
                values,
            )

    def add_ingestion_job(self, job_id: str, document_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (id, document_id, status, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, document_id, status, utcnow()),
            )

    def finish_ingestion_job(self, job_id: str, status: str, error_message: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, error_message, utcnow(), job_id),
            )

    def replace_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, text, page_number, section_title, token_count,
                    contains_pii, pii_types
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["id"],
                        document_id,
                        chunk["text"],
                        chunk.get("page_number"),
                        chunk.get("section_title"),
                        chunk["token_count"],
                        int(chunk["contains_pii"]),
                        json.dumps(chunk["pii_types"]),
                    )
                    for chunk in chunks
                ],
            )

    def list_documents(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE status != 'deleted' ORDER BY created_at DESC"
            ).fetchall()
        return rows

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ? AND status != 'deleted'",
                (document_id,),
            ).fetchone()
        return row

    def get_document_by_hash(self, source_hash: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source_hash = ? AND status != 'deleted'",
                (source_hash,),
            ).fetchone()
        return row

    def get_chunks(self, chunk_ids: list[str]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return rows

    def get_chunks_for_document(self, document_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        return rows

    def delete_document(self, document_id: str) -> sqlite3.Row | None:
        row = self.get_document(document_id)
        if row is None:
            return None
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute(
                "UPDATE documents SET status = 'deleted', updated_at = ? WHERE id = ?",
                (utcnow(), document_id),
            )
        return row

    def log_query(
        self,
        query_id: str,
        question: str,
        document_ids: list[str],
        retrieval_latency_ms: int,
        generation_latency_ms: int,
        model: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    id, question, document_ids, retrieval_latency_ms,
                    generation_latency_ms, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    question,
                    json.dumps(document_ids),
                    retrieval_latency_ms,
                    generation_latency_ms,
                    model,
                    utcnow(),
                ),
            )

    def log_guardrail_event(
        self,
        event_id: str,
        question: str,
        category: str,
        action: str,
        detail: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO guardrail_events (id, question, category, action, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, question, category, action, detail, utcnow()),
            )


db = Database(settings.database_path)
