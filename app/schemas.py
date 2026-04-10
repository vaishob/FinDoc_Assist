from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DocumentStatus = Literal["pending", "processing", "processed", "failed", "deleted"]
GuardrailStatus = Literal["allowed", "blocked", "masked"]


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class UploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    message: str


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = 0
    pii_detected: bool = False
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]


class DocumentDetailResponse(DocumentItem):
    summary: str | None = None
    error_message: str | None = None


class GuardrailResult(BaseModel):
    status: GuardrailStatus
    category: str | None = None
    message: str | None = None


class SourceSnippet(BaseModel):
    document_id: str
    page_number: int | None = None
    chunk_id: str
    score: float
    text: str
    contains_pii: bool = False
    pii_types: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_sources: bool = True


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[SourceSnippet]
    guardrail_result: GuardrailResult
    retrieval_latency_ms: int
    generation_latency_ms: int


class SummaryResponse(BaseModel):
    document_id: str
    summary: str | None


class ParsedPage(BaseModel):
    page_number: int | None = None
    text: str


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int | None = None
    section_title: str | None = None
    token_count: int
    contains_pii: bool = False
    pii_types: list[str] = Field(default_factory=list)


class RetrievedChunk(ChunkRecord):
    score: float
