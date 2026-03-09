"""Experiment runner: orchestrates evaluation, artifact generation, and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from libs.evaluation.dataset import load_dataset
from libs.evaluation.evaluator import RetrievalEvaluator, Retriever
from libs.evaluation.report import save_json_report, save_markdown_report
from libs.experiments.git_info import get_git_sha
from libs.experiments.models import ExperimentConfig, ExperimentRun, ExperimentStatus
from libs.experiments.store import ExperimentStore


class ExperimentRunner:
    """Orchestrates experiment execution.

    Steps:
    1. Load dataset from config path
    2. Run RetrievalEvaluator with the given retriever
    3. Save JSON and markdown artifacts
    4. Persist run to store (if provided)
    5. Return ExperimentRun with full provenance
    """

    def __init__(self, store: ExperimentStore | None = None) -> None:
        self._store = store

    def run(
        self,
        config: ExperimentConfig,
        retriever: Retriever,
        run_id: str = "",
    ) -> ExperimentRun:
        """Execute an experiment and return the completed run."""
        started_at = datetime.now(UTC)
        rid = run_id or f"exp-{started_at.strftime('%Y%m%d-%H%M%S')}"
        git_sha = get_git_sha()

        try:
            # Load dataset
            dataset_path = Path(config.dataset_path)
            cases = load_dataset(dataset_path)

            # Run evaluation
            evaluator = RetrievalEvaluator(
                config=config.eval_config,
                run_id=rid,
            )
            report = evaluator.evaluate(
                cases=cases,
                retriever=retriever,
                k_values=config.k_values,
            )

            # Save artifacts
            artifact_dir = Path(config.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            save_json_report(report, artifact_dir / f"{rid}.json")
            save_markdown_report(report, artifact_dir / f"{rid}.md")

            completed_at = datetime.now(UTC)
            duration = (completed_at - started_at).total_seconds()

            experiment_run = ExperimentRun(
                run_id=rid,
                config=config,
                report=report,
                status=ExperimentStatus.COMPLETED,
                started_at=started_at,
                completed_at=completed_at,
                git_sha=git_sha,
                duration_seconds=duration,
            )

        except Exception as exc:
            completed_at = datetime.now(UTC)
            duration = (completed_at - started_at).total_seconds()

            # Create a minimal failed report
            from libs.evaluation.models import EvaluationReport

            failed_report = EvaluationReport(
                run_id=rid,
                config=config.eval_config,
                query_results=[],
                stage_metrics=[],
                evaluated_at=completed_at,
            )

            experiment_run = ExperimentRun(
                run_id=rid,
                config=config,
                report=failed_report,
                status=ExperimentStatus.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                git_sha=git_sha,
                duration_seconds=duration,
                error=str(exc),
            )

        if self._store is not None:
            self._store.save(experiment_run)

        return experiment_run
