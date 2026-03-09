"""Async wrapper around any LexicalStore for async retrieval."""

from __future__ import annotations

from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.protocols import LexicalStore


class AsyncMemoryLexicalStore:
    """Async wrapper that delegates to a sync LexicalStore."""

    def __init__(
        self, inner: LexicalStore | None = None,
    ) -> None:
        self._inner: LexicalStore = inner or MemoryLexicalStore()

    @property
    def store_id(self) -> str:
        return self._inner.store_id

    @property
    def inner(self) -> LexicalStore:
        return self._inner

    async def async_search(
        self, query: RetrievalQuery,
    ) -> list[RetrievalCandidate]:
        """Delegate to sync search."""
        return self._inner.search(query)
