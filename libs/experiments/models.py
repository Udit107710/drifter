"""Experiment models: configs, runs, status, comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from libs.contracts.common import RunId
from libs.evaluation.models import EvaluationConfig, EvaluationReport


class ExperimentStatus(Enum):
    """Lifecycle status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ExperimentConfig:
    """Declarative experiment definition.

    Captures everything needed to reproduce an experiment:
    evaluation config, dataset location, artifact output directory,
    hypothesis, and optional baseline for comparison.
    """

    name: str
    hypothesis: str
    eval_config: EvaluationConfig
    dataset_path: str
    artifact_dir: str
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])
    baseline_run_id: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.hypothesis:
            raise ValueError("hypothesis must not be empty")
        if not self.dataset_path:
            raise ValueError("dataset_path must not be empty")
        if not self.artifact_dir:
            raise ValueError("artifact_dir must not be empty")


@dataclass(frozen=True)
class ExperimentRun:
    """Result of a completed (or failed) experiment execution.

    Captures full provenance: config snapshot, evaluation report,
    timing, git SHA, and any error.
    """

    run_id: RunId
    config: ExperimentConfig
    report: EvaluationReport
    status: ExperimentStatus
    started_at: datetime
    completed_at: datetime
    git_sha: str
    duration_seconds: float
    error: str | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class MetricDelta:
    """Change in a single metric between baseline and candidate."""

    metric_name: str
    baseline_value: float
    candidate_value: float
    absolute_change: float
    relative_change: float
    improved: bool


@dataclass(frozen=True)
class ExperimentComparison:
    """Side-by-side comparison of two experiment runs."""

    baseline: ExperimentRun
    candidate: ExperimentRun
    deltas: list[MetricDelta]
    config_diffs: dict[str, tuple[str, str]]
