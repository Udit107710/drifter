"""Sample script demonstrating experiment runner usage.

Creates a seed dataset, runs two experiments with different retrievers,
and compares the results.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from libs.evaluation.dataset import create_seed_dataset, save_dataset
from libs.experiments import (
    ExperimentRunner,
    InMemoryExperimentStore,
    compare_runs,
    generate_comparison_markdown,
)
from libs.experiments.sample_configs import (
    chunk_size_experiment,
    retrieval_mode_experiment,
)


class _PerfectRetriever:
    """Returns exact relevant chunk IDs for seed dataset queries."""

    def __init__(self) -> None:
        self._answers: dict[str, list[str]] = {
            "What is machine learning?": ["chunk-ml-001", "chunk-ml-002"],
            "How does backpropagation work?": [
                "chunk-bp-001", "chunk-bp-002", "chunk-bp-003",
            ],
            "What is a vector database?": ["chunk-vdb-001"],
            "Explain the transformer architecture": [
                "chunk-tf-001", "chunk-tf-002",
            ],
            "What is retrieval augmented generation?": [
                "chunk-rag-001", "chunk-rag-002", "chunk-rag-003",
            ],
        }

    def retrieve(self, query: str, k: int) -> list[str]:
        return self._answers.get(query, [])[:k]


class _WeakRetriever:
    """Returns partially relevant results."""

    def retrieve(self, query: str, k: int) -> list[str]:
        return ["chunk-ml-001", "irrelevant-1", "irrelevant-2"][:k]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Save seed dataset
        ds_path = base / "seed_dataset.json"
        save_dataset(create_seed_dataset(), ds_path)

        store = InMemoryExperimentStore()
        runner = ExperimentRunner(store=store)

        # Experiment 1: chunk size 256 with perfect retriever
        config_a = chunk_size_experiment(
            dataset_path=str(ds_path),
            artifact_dir=str(base / "exp_a"),
            chunk_size=256,
        )
        run_a = runner.run(config_a, _PerfectRetriever(), run_id="exp-chunk-256")
        print(f"Run A: {run_a.run_id} | status={run_a.status.value}")
        print(f"  Metrics: {run_a.report.aggregate_metrics}")

        # Experiment 2: retrieval mode dense with weak retriever
        config_b = retrieval_mode_experiment(
            dataset_path=str(ds_path),
            artifact_dir=str(base / "exp_b"),
            mode="dense",
        )
        run_b = runner.run(config_b, _WeakRetriever(), run_id="exp-dense-weak")
        print(f"Run B: {run_b.run_id} | status={run_b.status.value}")
        print(f"  Metrics: {run_b.report.aggregate_metrics}")

        # Compare
        comparison = compare_runs(run_b, run_a)
        md = generate_comparison_markdown(comparison)
        print("\n" + md)

        print(f"\nStore contains {len(store.list_all())} runs")


if __name__ == "__main__":
    main()
