"""Retrieval contracts: queries, candidates, and ranked results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from libs.contracts.chunks import Chunk
from libs.contracts.common import RetrievalMethod, TraceId


@dataclass(frozen=True)
class RetrievalQuery:
    """A structured query object for the retrieval broker.

    Wraps the raw user query with normalization results, trace context,
    and optional filters — so retrieval never operates on bare strings.
    """

    raw_query: str
    normalized_query: str
    trace_id: TraceId
    top_k: int = 50
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_query:
            raise ValueError("raw_query must not be empty")
        if not self.normalized_query:
            raise ValueError("normalized_query must not be empty")
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


@dataclass(frozen=True)
class RetrievalCandidate:
    """A chunk returned by a retrieval store with its relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: RetrievalMethod
    store_id: str

    def __post_init__(self) -> None:
        if not self.store_id:
            raise ValueError("store_id must not be empty")


@dataclass(frozen=True)
class RankedCandidate:
    """A retrieval candidate after reranking with a precision score."""

    candidate: RetrievalCandidate
    rank: int
    rerank_score: float
    reranker_id: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not self.reranker_id:
            raise ValueError("reranker_id must not be empty")
