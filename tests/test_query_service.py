import asyncio

from app.schemas import QueryRequest
from app.services.guardrails import GuardrailService
from app.services.query import QueryService


class FakeEmbeddingService:
    def embed_query(self, _text):
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def search(self, _vector, _top_k):
        return []


class FakeLLM:
    async def generate(self, _messages):
        raise AssertionError("LLM should not run when retrieval support is insufficient")


def test_query_blocks_when_no_context():
    service = QueryService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        guardrail_service=GuardrailService(),
        llm_client=FakeLLM(),
    )
    response = asyncio.run(service.answer(QueryRequest(question="What is the refund policy?")))
    assert response.guardrail_result.category == "insufficient_context"
