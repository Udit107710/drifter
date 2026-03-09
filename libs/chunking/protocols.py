"""Chunking protocols — contracts for chunking strategies and token counters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.chunks import Chunk
from libs.contracts.documents import CanonicalDocument


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a text string."""

    def count(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Splits a CanonicalDocument into a list of Chunks."""

    def chunk(self, doc: CanonicalDocument) -> list[Chunk]:
        """Split a document into chunks with full lineage."""
        ...

    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        ...
