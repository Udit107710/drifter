"""Reranking models: outcome, feature weights, and result envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from libs.contracts.retrieval import RankedCandidate, RetrievalQuery


class RerankerOutcome(Enum):
    """Outcome of a reranking run."""

    SUCCESS = "success"
    NO_CANDIDATES = "no_candidates"
    FAILED = "failed"


@dataclass(frozen=True)
class FeatureWeights:
    """Weights for each feature signal in the feature-based reranker."""

    retrieval_score: float = 1.0
    lexical_overlap: float = 0.3
    source_authority: float = 0.2
    freshness: float = 0.1
    title_match: float = 0.5
    source_type: float = 0.1
    source_reference: float = 2.0


@dataclass(frozen=True)
class RerankerResult:
    """Top-level return type of RerankerService.run()."""

    query: RetrievalQuery
    ranked_candidates: list[RankedCandidate]
    candidate_count: int
    outcome: RerankerOutcome
    reranker_id: str
    total_latency_ms: float
    completed_at: datetime
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
