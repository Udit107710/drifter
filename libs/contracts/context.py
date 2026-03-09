"""Context contracts: packed evidence for LLM generation."""

from __future__ import annotations

from dataclasses import dataclass

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId, SelectionReason, SourceId


@dataclass(frozen=True)
class ContextItem:
    """A single piece of evidence selected for the context pack."""

    chunk: Chunk
    rank: int
    token_count: int
    selection_reason: SelectionReason

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.token_count < 1:
            raise ValueError("token_count must be >= 1")


@dataclass(frozen=True)
class ContextPack:
    """The assembled evidence set passed to the generation stage.

    Contains selected chunks, total token accounting, and a diversity score
    indicating how well the evidence covers different sources/topics.
    """

    query: str
    evidence: list[ContextItem]
    total_tokens: int
    token_budget: int
    diversity_score: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("query must not be empty")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        if self.token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        if self.total_tokens > self.token_budget:
            raise ValueError("total_tokens must not exceed token_budget")
        if not (0.0 <= self.diversity_score <= 1.0):
            raise ValueError("diversity_score must be between 0.0 and 1.0")

    @property
    def chunk_ids(self) -> list[ChunkId]:
        """Chunk IDs included in this context pack, for tracing."""
        return [item.chunk.chunk_id for item in self.evidence]

    @property
    def source_ids(self) -> list[SourceId]:
        """Unique source IDs represented in this context pack."""
        return list(dict.fromkeys(item.chunk.source_id for item in self.evidence))
