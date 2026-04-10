from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from ..config import settings


@dataclass
class LLMResult:
    text: str
    model: str


class BaseLLMClient:
    async def generate(self, messages: list[dict[str, str]], max_tokens: int = 500) -> LLMResult:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    async def generate(self, messages: list[dict[str, str]], max_tokens: int = 500) -> LLMResult:
        if not settings.llm_base_url:
            raise RuntimeError("LLM_BASE_URL is not configured")
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResult(text=content, model=settings.llm_model)


class MockGroundedLLMClient(BaseLLMClient):
    async def generate(self, messages: list[dict[str, str]], max_tokens: int = 500) -> LLMResult:
        prompt = "\n".join(message["content"] for message in messages)
        answer = self._extract_answer(prompt)
        return LLMResult(text=answer, model="mock-grounded")

    def _extract_answer(self, prompt: str) -> str:
        context_match = re.search(r"Context:\n(?P<context>.*)", prompt, re.DOTALL)
        if not context_match:
            return "I do not have enough information in the uploaded documents."
        context = context_match.group("context").strip()
        first_line = next((line for line in context.splitlines() if line.strip()), "")
        if not first_line:
            return "I do not have enough information in the uploaded documents."
        excerpt = first_line
        if len(excerpt) > 320:
            excerpt = excerpt[:317] + "..."
        return f"Based on the retrieved context: {excerpt}"


class LLMClientFactory:
    def build(self) -> BaseLLMClient:
        if settings.llm_provider.lower() in {"mock", "none"}:
            return MockGroundedLLMClient()
        return OpenAICompatibleClient()
