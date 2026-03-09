"""Reranker protocol: the contract every reranker must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking retrieval candidates into a precision-oriented shortlist."""

    @property
    def reranker_id(self) -> str: ...

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]: ...
