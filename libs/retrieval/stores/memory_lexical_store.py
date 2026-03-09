"""In-memory lexical store for testing."""

from __future__ import annotations

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId, RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery


class MemoryLexicalStore:
    """Dict-backed lexical store using simple term matching.

    Scoring is based on the fraction of query terms found in the chunk content.
    Supports metadata filtering via query.filters.
    """

    def __init__(self, store_id: str = "memory-lexical") -> None:
        self._store_id = store_id
        self._chunks: dict[ChunkId, Chunk] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    def add(self, chunk: Chunk) -> None:
        """Index a chunk for lexical search."""
        self._chunks[chunk.chunk_id] = chunk

    def add_batch(self, chunks: list[Chunk]) -> None:
        """Index multiple chunks."""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        """Remove entries by chunk id."""
        count = 0
        for cid in chunk_ids:
            if cid in self._chunks:
                del self._chunks[cid]
                count += 1
        return count

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Search using term overlap scoring.

        Tokenizes the query and chunk content into lowercase terms.
        Score = (number of query terms found in chunk) / (total query terms).
        """
        query_terms = query.normalized_query.lower().split()
        if not query_terms:
            return []

        scored: list[tuple[float, Chunk]] = []

        for chunk in self._chunks.values():
            if not self._matches_filters(chunk, query.filters):
                continue
            chunk_terms = set(chunk.content.lower().split())
            matching = sum(1 for t in query_terms if t in chunk_terms)
            score = matching / len(query_terms)
            if score > 0.0:
                scored.append((score, chunk))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k
        top = scored[: query.top_k]

        return [
            RetrievalCandidate(
                chunk=chunk,
                score=score,
                retrieval_method=RetrievalMethod.LEXICAL,
                store_id=self._store_id,
            )
            for score, chunk in top
        ]

    def count(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, object]) -> bool:
        """Check if chunk metadata matches all filter key-value pairs."""
        if not filters:
            return True
        return all(chunk.metadata.get(key) == value for key, value in filters.items())
