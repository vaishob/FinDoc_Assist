from __future__ import annotations

from ..schemas import RetrievedChunk


class PromptBuilder:
    def build_messages(self, question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
        context_lines = []
        for index, chunk in enumerate(chunks, start=1):
            context_lines.append(
                f"{index}. [doc:{chunk.document_id}, page:{chunk.page_number or '?'}] {chunk.text}"
            )
        system_prompt = (
            "You are a document QA assistant for uploaded documents only.\n"
            "Answer only from the provided context.\n"
            "If the question is out of scope, unsupported by the documents, or requests restricted "
            "sensitive data, say so clearly.\n"
            "Do not use outside knowledge.\n"
            "Cite evidence as [doc:<id>, page:<n>]."
        )
        user_prompt = (
            f"User Question:\n{question}\n\n"
            f"Context:\n" + "\n".join(context_lines) + "\n\n"
            "Policy:\n"
            "- If evidence is insufficient, say that you do not have enough information in the uploaded documents.\n"
            "- If sensitive data is requested and policy forbids disclosure, refuse briefly and safely."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
