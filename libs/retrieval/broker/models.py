"""Retrieval broker models: configuration, results, and fused candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from libs.contracts.chunks import Chunk
from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery


class RetrievalMode(Enum):
    """Which retrieval backends the broker should use."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class BrokerConfig:
    """Constructor-time tuning knobs for the retrieval broker."""

    mode: RetrievalMode = RetrievalMode.HYBRID
    rrf_k: int = 60
    max_candidates_per_source: int = 0  # 0 = no cap
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    fanout_timeout_ms: int = 5000


class ErrorClassification(Enum):
    """Whether a retrieval error is transient (retryable) or permanent."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class StoreResult:
    """Per-store intermediate result for traceability."""

    store_id: str
    retrieval_method: RetrievalMethod
    candidates: list[RetrievalCandidate]
    candidate_count: int
    latency_ms: float
    error: str | None = None
    error_classification: ErrorClassification | None = None


@dataclass(frozen=True)
class FusedCandidate:
    """A candidate after RRF fusion with full provenance."""

    chunk: Chunk
    fused_score: float
    retrieval_method: RetrievalMethod
    contributing_stores: list[str]
    per_store_ranks: dict[str, int]
    per_store_scores: dict[str, float]


class BrokerOutcome(Enum):
    """Outcome of a broker retrieval run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_RESULTS = "no_results"


@dataclass(frozen=True)
class BrokerResult:
    """Top-level return type of RetrievalBroker.run()."""

    query: RetrievalQuery
    mode: RetrievalMode
    candidates: list[FusedCandidate]
    candidate_count: int
    store_results: list[StoreResult]
    outcome: BrokerOutcome
    total_latency_ms: float
    completed_at: datetime
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
