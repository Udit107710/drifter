"""Context builder models: config, outcome, exclusion records, and result envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from libs.contracts.common import ChunkId
from libs.contracts.context import ContextPack


class BuilderOutcome(Enum):
    SUCCESS = "success"
    EMPTY_CANDIDATES = "empty_candidates"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


@dataclass(frozen=True)
class ExclusionRecord:
    """Why a chunk was excluded from the context pack."""
    chunk_id: ChunkId
    reason: str
    token_count: int


@dataclass(frozen=True)
class BuilderConfig:
    """Configuration for context builder strategies."""
    token_budget: int = 3000
    diversity_weight: float = 0.3
    max_chunks: int = 0  # 0 = no limit
    deduplicate: bool = True


@dataclass(frozen=True)
class BuilderResult:
    """Top-level return type of ContextBuilderService.run()."""
    context_pack: ContextPack
    outcome: BuilderOutcome
    exclusions: list[ExclusionRecord]
    input_count: int
    dedup_removed: int
    total_latency_ms: float
    completed_at: datetime
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
