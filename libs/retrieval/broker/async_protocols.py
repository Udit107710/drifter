"""Async query embedder protocol and adapters for the retrieval broker."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.retrieval.broker.protocols import QueryEmbedder


@runtime_checkable
class AsyncQueryEmbedder(Protocol):
    """Async interface for embedding query text into a dense vector."""

    async def async_embed_query(self, text: str) -> list[float]:
        """Embed query text into a dense vector asynchronously."""
        ...


class SyncToAsyncEmbedder:
    """Wraps a sync ``QueryEmbedder`` to satisfy ``AsyncQueryEmbedder``.

    The underlying call is still synchronous — suitable for in-memory
    or fast I/O embedders.  For true async backends, implement
    ``AsyncQueryEmbedder`` natively.
    """

    def __init__(self, inner: QueryEmbedder) -> None:
        self._inner = inner

    async def async_embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)
