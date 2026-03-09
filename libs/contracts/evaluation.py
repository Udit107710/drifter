"""Evaluation contracts: ground truth cases and metric results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from libs.contracts.common import RunId


@dataclass(frozen=True)
class EvaluationCase:
    """A single ground-truth query-answer pair for evaluation.

    Contains the query, expected answer, and relevant document/chunk IDs
    so retrieval and answer quality can be measured.
    """

    case_id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: list[str]
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.query:
            raise ValueError("query must not be empty")
        if not self.expected_answer:
            raise ValueError("expected_answer must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty")


@dataclass(frozen=True)
class EvaluationResult:
    """Metric results for a single evaluation run.

    Records the experiment configuration alongside computed metrics
    so results are reproducible and comparable.
    """

    run_id: RunId
    case_id: str
    metrics: dict[str, float]
    config: dict[str, Any]
    evaluated_at: datetime
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
