"""Async query embedder protocol for the retrieval broker."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AsyncQueryEmbedder(Protocol):
    """Async interface for embedding query text into a dense vector."""

    async def async_embed_query(self, text: str) -> list[float]:
        """Embed query text into a dense vector asynchronously."""
        ...
