from __future__ import annotations

import json
import time
from uuid import uuid4

from ..config import settings
from ..db import db
from ..schemas import GuardrailResult, QueryRequest, QueryResponse, RetrievedChunk, SourceSnippet
from .guardrails import GuardrailService
from .prompting import PromptBuilder
from .vector_store import VectorStore


class QueryService:
    def __init__(self, embedding_service, vector_store: VectorStore, guardrail_service: GuardrailService, llm_client):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.guardrail_service = guardrail_service
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()

    async def answer(self, payload: QueryRequest) -> QueryResponse:
        question_guard = self.guardrail_service.evaluate_question(payload.question)
        if question_guard.status == "blocked":
            return QueryResponse(
                answer=None,
                sources=[],
                guardrail_result=GuardrailResult(
                    status=question_guard.status,
                    category=question_guard.category,
                    message=question_guard.message,
                ),
                retrieval_latency_ms=0,
                generation_latency_ms=0,
            )

        retrieval_start = time.perf_counter()
        query_vector = self.embedding_service.embed_query(payload.question)
        candidate_ids = self.vector_store.search(query_vector, payload.top_k or settings.top_k)
        chunk_rows = db.get_chunks([chunk_id for chunk_id, _score in candidate_ids])
        chunk_by_id = {row["id"]: row for row in chunk_rows}

        retrieved_chunks: list[RetrievedChunk] = []
        for chunk_id, score in candidate_ids:
            row = chunk_by_id.get(chunk_id)
            if row is None:
                continue
            if payload.document_ids and row["document_id"] not in payload.document_ids:
                continue
            pii_types = json.loads(row["pii_types"])
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    text=row["text"],
                    page_number=row["page_number"],
                    section_title=row["section_title"],
                    token_count=row["token_count"],
                    contains_pii=bool(row["contains_pii"]),
                    pii_types=pii_types,
                    score=score,
                )
            )
        support_guard = self.guardrail_service.evaluate_retrieval_support(
            payload.question,
            [chunk.score for chunk in retrieved_chunks],
        )
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)
        if support_guard.status == "blocked":
            return QueryResponse(
                answer=None,
                sources=[],
                guardrail_result=GuardrailResult(
                    status=support_guard.status,
                    category=support_guard.category,
                    message=support_guard.message,
                ),
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0,
            )

        final_chunks = self._finalize_context(retrieved_chunks)[: settings.final_context_k]
        prompt_messages = self.prompt_builder.build_messages(payload.question, final_chunks)
        generation_start = time.perf_counter()
        llm_result = await self.llm_client.generate(prompt_messages)
        generation_latency_ms = int((time.perf_counter() - generation_start) * 1000)

        masked_answer, status_override = self._mask_answer_if_needed(llm_result.text, final_chunks)
        sources = self._build_sources(final_chunks) if payload.include_sources else []
        guardrail_status = "masked" if status_override == "masked" else "allowed"

        db.log_query(
            query_id=f"query_{uuid4().hex[:12]}",
            question=payload.question,
            document_ids=payload.document_ids or [],
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            model=llm_result.model,
        )

        return QueryResponse(
            answer=masked_answer,
            sources=sources,
            guardrail_result=GuardrailResult(status=guardrail_status, category=None, message=None),
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )

    def _finalize_context(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        final: list[RetrievedChunk] = []
        for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            final.append(chunk)
            if len(final) >= settings.final_context_k:
                break
        return final

    def _mask_answer_if_needed(self, answer: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
        should_mask = any(chunk.contains_pii for chunk in chunks)
        pii_types: list[str] = []
        for chunk in chunks:
            pii_types.extend(chunk.pii_types)
        return self.guardrail_service.mask_if_needed(answer, should_mask, pii_types)

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[SourceSnippet]:
        sources: list[SourceSnippet] = []
        for chunk in chunks:
            text, _status = self.guardrail_service.mask_if_needed(
                chunk.text,
                chunk.contains_pii,
                chunk.pii_types,
            )
            sources.append(
                SourceSnippet(
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    score=round(chunk.score, 4),
                    text=text,
                    contains_pii=chunk.contains_pii,
                    pii_types=chunk.pii_types,
                )
            )
        return sources
