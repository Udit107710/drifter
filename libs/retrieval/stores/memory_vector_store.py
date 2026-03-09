"""In-memory vector store for testing."""

from __future__ import annotations

import math

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId, RetrievalMethod
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery


class MemoryVectorStore:
    """Dict-backed vector store using cosine similarity.

    Supports metadata filtering via query.filters. Each filter key is matched
    against chunk.metadata — a chunk passes if all filter key-value pairs match.
    """

    def __init__(self, store_id: str = "memory-vector") -> None:
        self._store_id = store_id
        self._vectors: dict[ChunkId, tuple[ChunkEmbedding, Chunk]] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
        """Index a chunk with its embedding vector."""
        self._vectors[chunk.chunk_id] = (embedding, chunk)

    def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
        """Index multiple chunks with their embeddings."""
        chunk_map = {c.chunk_id: c for c in chunks}
        for emb in embeddings:
            chunk = chunk_map.get(emb.chunk_id)
            if chunk is not None:
                self._vectors[emb.chunk_id] = (emb, chunk)

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        """Remove entries by chunk id."""
        count = 0
        for cid in chunk_ids:
            if cid in self._vectors:
                del self._vectors[cid]
                count += 1
        return count

    def search(
        self, query: RetrievalQuery, query_vector: list[float]
    ) -> list[RetrievalCandidate]:
        """Search using cosine similarity, with optional metadata filtering."""
        scored: list[tuple[float, Chunk]] = []

        for embedding, chunk in self._vectors.values():
            if not self._matches_filters(chunk, query.filters):
                continue
            score = self._cosine_similarity(query_vector, embedding.vector)
            scored.append((score, chunk))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k
        top = scored[: query.top_k]

        return [
            RetrievalCandidate(
                chunk=chunk,
                score=score,
                retrieval_method=RetrievalMethod.DENSE,
                store_id=self._store_id,
            )
            for score, chunk in top
        ]

    def count(self) -> int:
        return len(self._vectors)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, object]) -> bool:
        """Check if chunk metadata matches all filter key-value pairs."""
        if not filters:
            return True
        return all(chunk.metadata.get(key) == value for key, value in filters.items())
