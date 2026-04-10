from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import settings


class VectorStore:
    def __init__(self) -> None:
        self.base_path = Path(settings.vector_index_path)
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.ids_path = self.base_path.with_suffix(".ids.json")
        self.vectors_path = self.base_path.with_suffix(".vectors.npy")
        self._ids: list[str] = []
        self._vectors = np.empty((0, settings.embedding_dimension), dtype=np.float32)
        self._faiss = None
        self._load()

    def _load(self) -> None:
        if self.ids_path.exists():
            self._ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        if self.vectors_path.exists():
            self._vectors = np.load(self.vectors_path)
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(settings.embedding_dimension)
            if len(self._vectors):
                self._faiss.add(self._vectors)
        except Exception:
            self._faiss = None

    def _persist(self) -> None:
        self.ids_path.write_text(json.dumps(self._ids), encoding="utf-8")
        np.save(self.vectors_path, self._vectors)

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        existing = {chunk_id: idx for idx, chunk_id in enumerate(self._ids)}
        for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
            if chunk_id in existing:
                self._vectors[existing[chunk_id]] = vector
            else:
                self._ids.append(chunk_id)
                self._vectors = np.vstack([self._vectors, vector]) if len(self._vectors) else np.asarray([vector])
        self._rebuild_index()

    def delete_document_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids or not self._ids:
            return
        delete_ids = set(chunk_ids)
        keep_indices = [idx for idx, chunk_id in enumerate(self._ids) if chunk_id not in delete_ids]
        self._ids = [self._ids[idx] for idx in keep_indices]
        self._vectors = self._vectors[keep_indices] if keep_indices else np.empty(
            (0, settings.embedding_dimension), dtype=np.float32
        )
        self._rebuild_index()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not len(self._ids):
            return []
        query = np.asarray([query_vector], dtype=np.float32)
        if self._faiss is not None:
            scores, indices = self._faiss.search(query, min(top_k, len(self._ids)))
            return [
                (self._ids[idx], float(score))
                for idx, score in zip(indices[0], scores[0], strict=True)
                if idx >= 0
            ]
        scores = self._vectors @ query_vector
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._ids[idx], float(scores[idx])) for idx in ranked_indices]

    def _rebuild_index(self) -> None:
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(settings.embedding_dimension)
            if len(self._vectors):
                self._faiss.add(np.asarray(self._vectors, dtype=np.float32))
        except Exception:
            self._faiss = None
        self._persist()
