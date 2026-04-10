from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from ..config import settings
from ..db import db
from ..schemas import ChunkRecord
from .chunking import Chunker
from .guardrails import GuardrailService
from .parsing import DocumentParser
from .vector_store import VectorStore


class IngestionService:
    def __init__(self, embedding_service, vector_store: VectorStore, guardrail_service: GuardrailService):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.guardrail_service = guardrail_service
        self.parser = DocumentParser()
        self.chunker = Chunker()
        settings.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> str:
        await file.seek(0)
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_mb:
            raise HTTPException(status_code=413, detail="File exceeds maximum allowed size.")
        content_type = file.content_type or ""
        suffix = Path(file.filename or "upload.txt").suffix.lower()
        if content_type not in {"application/pdf", "text/plain"} and suffix not in {".pdf", ".txt"}:
            raise HTTPException(status_code=400, detail="Only PDF and text documents are supported.")
        source_hash = hashlib.sha256(content).hexdigest()
        existing = db.get_document_by_hash(source_hash)
        if existing is not None:
            raise HTTPException(status_code=409, detail="This document was already uploaded.")
        document_id = f"doc_{uuid4().hex[:12]}"
        safe_name = f"{document_id}{suffix or '.bin'}"
        stored_path = settings.upload_dir / safe_name
        stored_path.write_bytes(content)
        db.create_document(
            document_id=document_id,
            filename=file.filename or safe_name,
            stored_path=str(stored_path),
            content_type=content_type or ("application/pdf" if suffix == ".pdf" else "text/plain"),
            source_hash=source_hash,
        )
        return document_id

    def process_document(self, document_id: str) -> None:
        job_id = f"job_{uuid4().hex[:12]}"
        db.add_ingestion_job(job_id, document_id, "processing")
        db.update_document_status(document_id, "processing")
        record = db.get_document(document_id)
        if record is None:
            db.finish_ingestion_job(job_id, "failed", "Document not found.")
            return
        try:
            pages = self.parser.parse(Path(record["stored_path"]), record["content_type"])
            chunks = self.chunker.chunk(document_id, pages)
            annotated = [self._annotate_chunk(chunk) for chunk in chunks]
            vectors = self.embedding_service.embed_texts([chunk.text for chunk in annotated])
            self.vector_store.upsert([chunk.chunk_id for chunk in annotated], vectors)
            db.replace_chunks(document_id, [self._chunk_to_row(chunk) for chunk in annotated])
            summary = self._build_summary(annotated)
            db.update_document_status(
                document_id,
                "processed",
                chunk_count=len(annotated),
                pii_detected=any(chunk.contains_pii for chunk in annotated),
                summary=summary,
            )
            db.finish_ingestion_job(job_id, "processed")
        except Exception as exc:
            db.update_document_status(document_id, "failed", error_message=str(exc))
            db.finish_ingestion_job(job_id, "failed", str(exc))

    def delete_document(self, document_id: str) -> bool:
        record = db.get_document(document_id)
        if record is None:
            return False
        chunk_rows = db.get_chunks_for_document(document_id)
        self.vector_store.delete_document_chunks([row["id"] for row in chunk_rows])
        path = Path(record["stored_path"])
        if path.exists():
            path.unlink()
        db.delete_document(document_id)
        return True

    def _annotate_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        contains_pii, pii_types = self.guardrail_service.detect_pii(chunk.text)
        chunk.contains_pii = contains_pii
        chunk.pii_types = pii_types
        return chunk

    @staticmethod
    def _chunk_to_row(chunk: ChunkRecord) -> dict:
        return {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "page_number": chunk.page_number,
            "section_title": chunk.section_title,
            "token_count": chunk.token_count,
            "contains_pii": chunk.contains_pii,
            "pii_types": chunk.pii_types,
        }

    @staticmethod
    def _build_summary(chunks: list[ChunkRecord]) -> str | None:
        if not chunks:
            return None
        summary_parts = [chunk.text for chunk in chunks[:3]]
        summary = " ".join(summary_parts)
        return summary[:700] + ("..." if len(summary) > 700 else "")
