"""In-memory embedding repository for testing."""

from __future__ import annotations

from collections import defaultdict

from libs.contracts.common import ChunkId, EmbeddingId
from libs.contracts.embeddings import ChunkEmbedding


class MemoryEmbeddingRepository:
    """Dict-backed embedding store with secondary indexes."""

    def __init__(self) -> None:
        self._store: dict[EmbeddingId, ChunkEmbedding] = {}
        self._by_chunk: dict[ChunkId, list[EmbeddingId]] = defaultdict(list)
        self._by_chunk_model: dict[tuple[ChunkId, str, str], EmbeddingId] = {}

    def store(self, embedding: ChunkEmbedding) -> None:
        key = (embedding.chunk_id, embedding.model_id, embedding.model_version)
        # Upsert: remove old entry for same chunk+model combo if exists
        old_id = self._by_chunk_model.get(key)
        if old_id is not None:
            self._store.pop(old_id, None)
            chunk_list = self._by_chunk.get(embedding.chunk_id, [])
            if old_id in chunk_list:
                chunk_list.remove(old_id)

        self._store[embedding.embedding_id] = embedding
        self._by_chunk[embedding.chunk_id].append(embedding.embedding_id)
        self._by_chunk_model[key] = embedding.embedding_id

    def store_batch(self, embeddings: list[ChunkEmbedding]) -> None:
        for emb in embeddings:
            self.store(emb)

    def get_by_chunk(self, chunk_id: ChunkId) -> list[ChunkEmbedding]:
        emb_ids = self._by_chunk.get(chunk_id, [])
        return [self._store[eid] for eid in emb_ids if eid in self._store]

    def get_by_chunk_and_model(
        self, chunk_id: ChunkId, model_id: str, model_version: str
    ) -> ChunkEmbedding | None:
        emb_id = self._by_chunk_model.get((chunk_id, model_id, model_version))
        if emb_id is None:
            return None
        return self._store.get(emb_id)

    def list_by_model(self, model_id: str, model_version: str) -> list[ChunkEmbedding]:
        return [
            emb
            for emb in self._store.values()
            if emb.model_id == model_id and emb.model_version == model_version
        ]

    def delete_by_chunk(self, chunk_id: ChunkId) -> int:
        emb_ids = self._by_chunk.pop(chunk_id, [])
        count = 0
        for eid in emb_ids:
            if eid in self._store:
                emb = self._store.pop(eid)
                key = (emb.chunk_id, emb.model_id, emb.model_version)
                self._by_chunk_model.pop(key, None)
                count += 1
        return count
