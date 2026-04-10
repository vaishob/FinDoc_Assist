from __future__ import annotations

import re
from typing import Iterable

from ..schemas import ChunkRecord, ParsedPage


class Chunker:
    def __init__(self, target_words: int = 180, overlap_words: int = 40):
        self.target_words = target_words
        self.overlap_words = overlap_words

    def chunk(self, document_id: str, pages: Iterable[ParsedPage]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        chunk_index = 0
        for page in pages:
            cleaned = self._normalize(page.text)
            if not cleaned:
                continue
            paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
            word_buffer: list[str] = []
            for paragraph in paragraphs:
                paragraph_words = paragraph.split()
                if len(word_buffer) + len(paragraph_words) <= self.target_words:
                    word_buffer.extend(paragraph_words)
                    continue
                if word_buffer:
                    chunks.append(
                        ChunkRecord(
                            chunk_id=f"{document_id}_chunk_{chunk_index}",
                            document_id=document_id,
                            text=" ".join(word_buffer),
                            page_number=page.page_number,
                            token_count=len(word_buffer),
                        )
                    )
                    chunk_index += 1
                    word_buffer = word_buffer[-self.overlap_words :]
                for sub_chunk in self._split_large_paragraph(
                    document_id=document_id,
                    page_number=page.page_number,
                    words=paragraph_words,
                    chunk_index_start=chunk_index,
                ):
                    chunks.append(sub_chunk)
                    chunk_index += 1
                word_buffer = []
            if word_buffer:
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        document_id=document_id,
                        text=" ".join(word_buffer),
                        page_number=page.page_number,
                        token_count=len(word_buffer),
                    )
                )
                chunk_index += 1
        return chunks

    def _split_large_paragraph(
        self,
        document_id: str,
        page_number: int | None,
        words: list[str],
        chunk_index_start: int,
    ) -> list[ChunkRecord]:
        output: list[ChunkRecord] = []
        cursor = 0
        local_index = chunk_index_start
        while cursor < len(words):
            segment = words[cursor : cursor + self.target_words]
            output.append(
                ChunkRecord(
                    chunk_id=f"{document_id}_chunk_{local_index}",
                    document_id=document_id,
                    text=" ".join(segment),
                    page_number=page_number,
                    token_count=len(segment),
                )
            )
            local_index += 1
            if cursor + self.target_words >= len(words):
                break
            cursor += self.target_words - self.overlap_words
        return output

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
