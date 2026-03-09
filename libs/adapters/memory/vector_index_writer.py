"""In-memory vector index writer for testing."""

from __future__ import annotations

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId
from libs.contracts.embeddings import ChunkEmbedding


class MemoryVectorIndexWriter:
    """Dict-backed vector index with upsert semantics."""

    def __init__(self) -> None:
        self._store: dict[ChunkId, tuple[ChunkEmbedding, Chunk]] = {}

    def write_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> int:
        chunk_map = {c.chunk_id: c for c in chunks}
        count = 0
        for emb in embeddings:
            chunk = chunk_map.get(emb.chunk_id)
            if chunk is not None:
                self._store[emb.chunk_id] = (emb, chunk)
                count += 1
        return count

    def delete_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int:
        count = 0
        for cid in chunk_ids:
            if cid in self._store:
                del self._store[cid]
                count += 1
        return count
