"""Deduplication: remove chunks with identical content_hash."""

from __future__ import annotations

from libs.context_builder.models import ExclusionRecord
from libs.contracts.retrieval import RankedCandidate


def deduplicate(
    candidates: list[RankedCandidate],
) -> tuple[list[RankedCandidate], list[ExclusionRecord]]:
    """Remove duplicate chunks by content_hash, keeping the first (highest-ranked) occurrence."""
    seen: set[str] = set()
    kept: list[RankedCandidate] = []
    excluded: list[ExclusionRecord] = []

    for rc in candidates:
        h = rc.candidate.chunk.content_hash
        if h in seen:
            excluded.append(
                ExclusionRecord(
                    chunk_id=rc.candidate.chunk.chunk_id,
                    reason=f"duplicate:content_hash={h}",
                    token_count=rc.candidate.chunk.token_count,
                )
            )
        else:
            seen.add(h)
            kept.append(rc)

    return kept, excluded
