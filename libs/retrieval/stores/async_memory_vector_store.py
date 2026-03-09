"""Async wrapper around any VectorStore for async retrieval."""

from __future__ import annotations

from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore
from libs.retrieval.stores.protocols import VectorStore


class AsyncMemoryVectorStore:
    """Async wrapper that delegates to a sync VectorStore."""

    def __init__(
        self, inner: VectorStore | None = None,
    ) -> None:
        self._inner: VectorStore = inner or MemoryVectorStore()

    @property
    def store_id(self) -> str:
        return self._inner.store_id

    @property
    def inner(self) -> VectorStore:
        return self._inner

    async def async_search(
        self, query: RetrievalQuery, query_vector: list[float],
    ) -> list[RetrievalCandidate]:
        """Delegate to sync search."""
        return self._inner.search(query, query_vector)
