"""Reciprocal Rank Fusion (RRF) for merging ranked retrieval lists."""

from __future__ import annotations

from libs.contracts.common import ChunkId, RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate
from libs.retrieval.broker.models import FusedCandidate


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalCandidate]],
    weights: list[float],
    k: int = 60,
) -> list[FusedCandidate]:
    """Fuse multiple ranked candidate lists using RRF.

    For each unique chunk across all lists:
        fused_score = sum(weight_i / (k + rank_i))

    Args:
        ranked_lists: Lists of candidates, each pre-sorted by descending score.
        weights: Weight multiplier for each list (1:1 correspondence).
        k: RRF constant (default 60). Higher k flattens rank differences.

    Returns:
        Fused candidates sorted by descending fused_score.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("ranked_lists and weights must have the same length")

    # Accumulate per-chunk data
    # chunk_id -> first seen candidate (for the Chunk object)
    chunks: dict[ChunkId, RetrievalCandidate] = {}
    scores: dict[ChunkId, float] = {}
    stores: dict[ChunkId, list[str]] = {}
    ranks: dict[ChunkId, dict[str, int]] = {}
    raw_scores: dict[ChunkId, dict[str, float]] = {}
    methods: dict[ChunkId, set[RetrievalMethod]] = {}

    for candidates, weight in zip(ranked_lists, weights, strict=True):
        for rank, candidate in enumerate(candidates, start=1):
            cid = candidate.chunk.chunk_id
            rrf_contribution = weight / (k + rank)

            if cid not in chunks:
                chunks[cid] = candidate
                scores[cid] = 0.0
                stores[cid] = []
                ranks[cid] = {}
                raw_scores[cid] = {}
                methods[cid] = set()

            scores[cid] += rrf_contribution
            if candidate.store_id not in stores[cid]:
                stores[cid].append(candidate.store_id)
            ranks[cid][candidate.store_id] = rank
            raw_scores[cid][candidate.store_id] = candidate.score
            methods[cid].add(candidate.retrieval_method)

    # Build fused candidates
    fused: list[FusedCandidate] = []
    for cid, candidate in chunks.items():
        # Determine method: HYBRID if multiple methods contributed
        method_set = methods[cid]
        method = (
            RetrievalMethod.HYBRID if len(method_set) > 1 else next(iter(method_set))
        )

        fused.append(
            FusedCandidate(
                chunk=candidate.chunk,
                fused_score=scores[cid],
                retrieval_method=method,
                contributing_stores=stores[cid],
                per_store_ranks=ranks[cid],
                per_store_scores=raw_scores[cid],
            )
        )

    # Sort by fused_score descending, tie-break by max individual score descending
    fused.sort(
        key=lambda fc: (fc.fused_score, max(fc.per_store_scores.values(), default=0.0)),
        reverse=True,
    )

    return fused
