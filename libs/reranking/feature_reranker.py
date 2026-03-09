"""FeatureBasedReranker: blends multiple signals into a composite rerank score."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery
from libs.reranking.models import FeatureWeights


class FeatureBasedReranker:
    """Reranker that computes a weighted sum of normalized feature signals.

    Features (each normalized to [0, 1]):
    1. retrieval_score — min-max normalized across the batch
    2. lexical_overlap — fraction of query terms in chunk content
    3. source_authority — from chunk.metadata["authority"], clamped
    4. freshness — exp(-age_days / 365) from chunk.metadata["updated_at"]
    5. title_match — 1.0 if query is substring of chunk.metadata["title"]
    6. source_type — from chunk.metadata["source_type_score"], clamped
    7. source_reference — 1.0 if query references a chapter/section that
       matches the chunk's source_id (e.g. "chapter 20" → "chapter_20")
    """

    def __init__(
        self,
        weights: FeatureWeights | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        self._weights = weights or FeatureWeights()
        self._reference_time = reference_time or datetime.now(UTC)

    @property
    def reranker_id(self) -> str:
        return "feature-based-v1"

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        # Pre-compute min/max scores for normalization
        scores = [c.score for c in candidates]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        # Tokenize query for lexical overlap
        query_terms = set(re.findall(r"\w+", query.normalized_query.lower()))

        # Extract source reference patterns (e.g. "chapter 20" → "chapter_20")
        source_ref_pattern = _extract_source_reference(query.normalized_query)

        scored: list[tuple[float, float, RetrievalCandidate]] = []
        for c in candidates:
            final = self._compute_score(
                c, query, query_terms, min_score, score_range, source_ref_pattern,
            )
            scored.append((final, c.score, c))

        # Sort by final score descending, tie-break by retrieval score descending
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        return [
            RankedCandidate(
                candidate=c,
                rank=i + 1,
                rerank_score=final,
                reranker_id=self.reranker_id,
            )
            for i, (final, _, c) in enumerate(scored)
        ]

    def score_breakdown(
        self,
        candidate: RetrievalCandidate,
        query: RetrievalQuery,
        min_score: float,
        score_range: float,
    ) -> dict[str, float]:
        """Return per-feature scores for a single candidate (rag_trace_analysis skill).

        Useful for debugging why a candidate was ranked where it was.
        """
        query_terms = set(re.findall(r"\w+", query.normalized_query.lower()))
        w = self._weights
        meta = candidate.chunk.metadata

        retrieval_norm = (candidate.score - min_score) / score_range if score_range > 0 else 1.0

        if query_terms:
            content_terms = set(re.findall(r"\w+", candidate.chunk.content.lower()))
            lexical_overlap = len(query_terms & content_terms) / len(query_terms)
        else:
            lexical_overlap = 0.0

        authority = _clamp(float(meta.get("authority", 0.5)))

        updated_at = meta.get("updated_at")
        if isinstance(updated_at, datetime):
            ref = self._reference_time
            if updated_at.tzinfo is None:
                ref = ref.replace(tzinfo=None)
            age_days = max((ref - updated_at).days, 0)
            freshness = math.exp(-age_days / 365)
        else:
            freshness = 0.5

        title = str(meta.get("title", "")).lower()
        title_match = 1.0 if query.normalized_query.lower() in title else 0.0

        source_type = _clamp(float(meta.get("source_type_score", 0.5)))

        source_ref_pattern = _extract_source_reference(query.normalized_query)
        source_ref = 0.0
        if source_ref_pattern:
            source_id = candidate.chunk.source_id.lower()
            content_lower = candidate.chunk.content[:200].lower()
            if source_ref_pattern in source_id or source_ref_pattern in content_lower:
                source_ref = 1.0

        return {
            "retrieval_score": retrieval_norm,
            "retrieval_score_weighted": w.retrieval_score * retrieval_norm,
            "lexical_overlap": lexical_overlap,
            "lexical_overlap_weighted": w.lexical_overlap * lexical_overlap,
            "source_authority": authority,
            "source_authority_weighted": w.source_authority * authority,
            "freshness": freshness,
            "freshness_weighted": w.freshness * freshness,
            "title_match": title_match,
            "title_match_weighted": w.title_match * title_match,
            "source_type": source_type,
            "source_type_weighted": w.source_type * source_type,
            "source_reference": source_ref,
            "source_reference_weighted": w.source_reference * source_ref,
        }

    def _compute_score(
        self,
        candidate: RetrievalCandidate,
        query: RetrievalQuery,
        query_terms: set[str],
        min_score: float,
        score_range: float,
        source_ref_pattern: str = "",
    ) -> float:
        w = self._weights
        meta = candidate.chunk.metadata

        # 1. Retrieval score (min-max normalized)
        retrieval_norm = (candidate.score - min_score) / score_range if score_range > 0 else 1.0

        # 2. Lexical overlap
        if query_terms:
            content_terms = set(re.findall(r"\w+", candidate.chunk.content.lower()))
            lexical_overlap = len(query_terms & content_terms) / len(query_terms)
        else:
            lexical_overlap = 0.0

        # 3. Source authority
        authority = _clamp(float(meta.get("authority", 0.5)))

        # 4. Freshness
        updated_at = meta.get("updated_at")
        if isinstance(updated_at, datetime):
            ref = self._reference_time
            if updated_at.tzinfo is None:
                ref = ref.replace(tzinfo=None)
            age_days = max((ref - updated_at).days, 0)
            freshness = math.exp(-age_days / 365)
        else:
            freshness = 0.5

        # 5. Title match
        title = str(meta.get("title", "")).lower()
        title_match = 1.0 if query.normalized_query.lower() in title else 0.0

        # 6. Source type
        source_type = _clamp(float(meta.get("source_type_score", 0.5)))

        # 7. Source reference (chapter/section match)
        source_ref = 0.0
        if source_ref_pattern:
            source_id = candidate.chunk.source_id.lower()
            content_lower = candidate.chunk.content[:200].lower()
            if source_ref_pattern in source_id or source_ref_pattern in content_lower:
                source_ref = 1.0

        return (
            w.retrieval_score * retrieval_norm
            + w.lexical_overlap * lexical_overlap
            + w.source_authority * authority
            + w.freshness * freshness
            + w.title_match * title_match
            + w.source_type * source_type
            + w.source_reference * source_ref
        )


def _extract_source_reference(query: str) -> str:
    """Extract a source reference pattern from a query.

    Detects patterns like "chapter 20", "section 3", "part 5" and returns
    a normalized form suitable for matching against source_id or content
    (e.g. "chapter_20", "section_3").

    Returns empty string if no reference is found.
    """
    match = re.search(
        r"\b(chapter|section|part|appendix)\s+(\d+)\b",
        query,
        re.IGNORECASE,
    )
    if match:
        label = match.group(1).lower()
        number = match.group(2)
        return f"{label}_{number}"
    return ""


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))
