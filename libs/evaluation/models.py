"""Evaluation models: reports, stage metrics, per-query breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from libs.contracts.common import RunId


@dataclass(frozen=True)
class QueryResult:
    """Per-query evaluation result with metric breakdown."""
    case_id: str
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageMetrics:
    """Aggregate metrics for a single evaluation stage (retrieval or answer)."""
    stage: str
    metric_means: dict[str, float]
    metric_medians: dict[str, float] = field(default_factory=dict)
    query_count: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration snapshot for reproducibility."""
    retrieval_mode: str = ""
    embedding_model: str = ""
    reranker_id: str = ""
    chunking_strategy: str = ""
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    """Complete evaluation report with per-query and aggregate results."""
    run_id: RunId
    config: EvaluationConfig
    query_results: list[QueryResult]
    stage_metrics: list[StageMetrics]
    evaluated_at: datetime
    dataset_name: str = ""
    dataset_size: int = 0
    schema_version: int = 1

    @property
    def aggregate_metrics(self) -> dict[str, float]:
        """Flat dict of all stage metrics for quick access."""
        result: dict[str, float] = {}
        for sm in self.stage_metrics:
            for key, val in sm.metric_means.items():
                result[f"{sm.stage}.{key}"] = val
        return result
