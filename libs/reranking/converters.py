"""Converters: bridge FusedCandidate from the retrieval broker to RetrievalCandidate."""

from __future__ import annotations

from libs.contracts.retrieval import RetrievalCandidate
from libs.retrieval.broker.models import FusedCandidate


def fused_to_retrieval_candidate(fc: FusedCandidate) -> RetrievalCandidate:
    """Convert a single FusedCandidate to a RetrievalCandidate.

    Maps fused_score -> score, first contributing_stores entry -> store_id.
    """
    return RetrievalCandidate(
        chunk=fc.chunk,
        score=fc.fused_score,
        retrieval_method=fc.retrieval_method,
        store_id=fc.contributing_stores[0],
    )


def fused_list_to_retrieval_candidates(
    fused: list[FusedCandidate],
) -> list[RetrievalCandidate]:
    """Convert a list of FusedCandidates to RetrievalCandidates."""
    return [fused_to_retrieval_candidate(fc) for fc in fused]
