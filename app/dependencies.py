from __future__ import annotations

from functools import lru_cache

from .services.embedding import EmbeddingService
from .services.guardrails import GuardrailService
from .services.ingestion import IngestionService
from .services.llm import LLMClientFactory
from .services.query import QueryService
from .services.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_guardrail_service() -> GuardrailService:
    return GuardrailService()


@lru_cache
def get_llm_client():
    return LLMClientFactory().build()


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        guardrail_service=get_guardrail_service(),
    )


@lru_cache
def get_query_service() -> QueryService:
    return QueryService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        guardrail_service=get_guardrail_service(),
        llm_client=get_llm_client(),
    )
