"""Retrieval store protocols: vector and lexical search interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery


@runtime_checkable
class VectorStore(Protocol):
    """Dense (embedding-based) retrieval backend.

    A single pluggable interface covering both the write path (populating
    the index) and the read path (searching it).  Any implementation —
    in-memory, Qdrant, OpenSearch k-NN — must satisfy both sides so
    callers can program against ``VectorStore`` without knowing the
    concrete backend.
    """

    @property
    def store_id(self) -> str: ...

    # ── write path ───────────────────────────────────────────────────

    def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
        """Index a single chunk with its embedding vector."""
        ...

    def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
        """Index multiple chunks with their embeddings."""
        ...

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        """Remove entries by chunk id.  Returns the number deleted."""
        ...

    # ── read path ────────────────────────────────────────────────────

    def search(self, query: RetrievalQuery, query_vector: list[float]) -> list[RetrievalCandidate]:
        """Search the vector index using the provided query vector.

        Args:
            query: Structured retrieval query with top_k and filters.
            query_vector: Dense vector representation of the query.

        Returns:
            Candidates sorted by descending similarity score.
        """
        ...

    def count(self) -> int:
        """Return the number of indexed items."""
        ...


@runtime_checkable
class LexicalStore(Protocol):
    """Full-text / keyword-based retrieval backend.

    A single pluggable interface covering both the write path (populating
    the index) and the read path (searching it).
    """

    @property
    def store_id(self) -> str: ...

    # ── write path ───────────────────────────────────────────────────

    def add(self, chunk: Chunk) -> None:
        """Index a single chunk for lexical search."""
        ...

    def add_batch(self, chunks: list[Chunk]) -> None:
        """Index multiple chunks."""
        ...

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        """Remove entries by chunk id.  Returns the number deleted."""
        ...

    # ── read path ────────────────────────────────────────────────────

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Search the lexical index using the query's normalized text.

        Args:
            query: Structured retrieval query with top_k and filters.

        Returns:
            Candidates sorted by descending relevance score.
        """
        ...

    def count(self) -> int:
        """Return the number of indexed items."""
        ...
