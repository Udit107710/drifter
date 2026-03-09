"""Source-cap filtering for retrieval broker candidates."""

from __future__ import annotations

from collections import defaultdict

from libs.retrieval.broker.models import FusedCandidate


def apply_source_caps(
    candidates: list[FusedCandidate],
    max_per_source: int,
) -> list[FusedCandidate]:
    """Limit candidates per source_id for diversity.

    Iterates candidates in order (assumed pre-sorted by fused_score),
    counts per source_id, and skips any exceeding max_per_source.

    Args:
        candidates: Fused candidates sorted by descending score.
        max_per_source: Maximum candidates from any single source. 0 = no cap.

    Returns:
        Filtered candidate list preserving input order.
    """
    if max_per_source <= 0:
        return candidates

    counts: dict[str, int] = defaultdict(int)
    result: list[FusedCandidate] = []

    for candidate in candidates:
        source_id = candidate.chunk.source_id
        if counts[source_id] < max_per_source:
            result.append(candidate)
            counts[source_id] += 1

    return result
