"""Experiment management subsystem.

Responsibilities:
- Define reproducible experiment configurations
- Orchestrate experiment execution with full provenance
- Persist and query experiment runs
- Compare runs with metric deltas and config diffs
"""

from libs.experiments.comparison import compare_runs, generate_comparison_markdown
from libs.experiments.git_info import get_git_sha
from libs.experiments.models import (
    ExperimentComparison,
    ExperimentConfig,
    ExperimentRun,
    ExperimentStatus,
    MetricDelta,
)
from libs.experiments.runner import ExperimentRunner
from libs.experiments.sample_configs import (
    chunk_size_experiment,
    reranker_experiment,
    retrieval_mode_experiment,
)
from libs.experiments.store import ExperimentStore, InMemoryExperimentStore

__all__ = [
    "ExperimentComparison",
    "ExperimentConfig",
    "ExperimentRun",
    "ExperimentRunner",
    "ExperimentStatus",
    "ExperimentStore",
    "InMemoryExperimentStore",
    "MetricDelta",
    "chunk_size_experiment",
    "compare_runs",
    "generate_comparison_markdown",
    "get_git_sha",
    "reranker_experiment",
    "retrieval_mode_experiment",
]
