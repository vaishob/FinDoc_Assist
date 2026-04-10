from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from ..config import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.dimension = settings.embedding_dimension
        self._model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(settings.embedding_model)
        except Exception:
            return None

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self._model is not None:
            vectors = self._model.encode(text_list, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        return np.asarray([self._hash_embed(text) for text in text_list], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(0, len(digest), 2):
                bucket = int.from_bytes(digest[idx : idx + 2], "little") % self.dimension
                vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector
