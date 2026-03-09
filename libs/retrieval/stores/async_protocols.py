"""Async retrieval store protocols: vector and lexical search interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery


@runtime_checkable
class AsyncVectorStore(Protocol):
    """Async dense (embedding-based) retrieval backend."""

    @property
    def store_id(self) -> str: ...

    async def async_search(
        self, query: RetrievalQuery, query_vector: list[float],
    ) -> list[RetrievalCandidate]:
        """Search the vector index asynchronously."""
        ...


@runtime_checkable
class AsyncLexicalStore(Protocol):
    """Async full-text / keyword-based retrieval backend."""

    @property
    def store_id(self) -> str: ...

    async def async_search(
        self, query: RetrievalQuery,
    ) -> list[RetrievalCandidate]:
        """Search the lexical index asynchronously."""
        ...
