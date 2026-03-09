"""Tests for the experiments subsystem."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from libs.contracts.evaluation import EvaluationCase
from libs.evaluation.dataset import save_dataset
from libs.evaluation.models import (
    EvaluationConfig,
    EvaluationReport,
    QueryResult,
    StageMetrics,
)
from libs.experiments.comparison import compare_runs, generate_comparison_markdown
from libs.experiments.models import (
    ExperimentConfig,
    ExperimentRun,
    ExperimentStatus,
)
from libs.experiments.runner import ExperimentRunner
from libs.experiments.sample_configs import (
    chunk_size_experiment,
    reranker_experiment,
    retrieval_mode_experiment,
)
from libs.experiments.store import InMemoryExperimentStore

# ── Helpers ──────────────────────────────────────────────────────────


def _make_eval_config(**kwargs: object) -> EvaluationConfig:
    defaults = {
        "retrieval_mode": "dense",
        "embedding_model": "test-model",
        "reranker_id": "none",
        "chunking_strategy": "fixed-512",
    }
    defaults.update(kwargs)
    return EvaluationConfig(**defaults)  # type: ignore[arg-type]


def _make_experiment_config(
    name: str = "test-exp",
    hypothesis: str = "Testing hypothesis",
    dataset_path: str = "/tmp/ds.json",
    artifact_dir: str = "/tmp/artifacts",
    **kwargs: object,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        hypothesis=hypothesis,
        eval_config=kwargs.pop("eval_config", _make_eval_config()),  # type: ignore[arg-type]
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        **kwargs,  # type: ignore[arg-type]
    )


def _make_report(
    run_id: str = "run-1",
    metrics: dict[str, float] | None = None,
) -> EvaluationReport:
    m = metrics or {"recall@5": 0.8, "precision@5": 0.6, "mrr": 0.7}
    return EvaluationReport(
        run_id=run_id,
        config=_make_eval_config(),
        query_results=[
            QueryResult(
                case_id="c1",
                query="test query",
                retrieved_ids=["a", "b"],
                relevant_ids=["a"],
                metrics=m,
            ),
        ],
        stage_metrics=[
            StageMetrics(
                stage="retrieval",
                metric_means=m,
                query_count=1,
            ),
        ],
        evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
        dataset_size=1,
    )


def _make_run(
    run_id: str = "run-1",
    name: str = "test-exp",
    metrics: dict[str, float] | None = None,
    tags: list[str] | None = None,
    eval_config: EvaluationConfig | None = None,
    started_at: datetime | None = None,
) -> ExperimentRun:
    cfg_kwargs: dict[str, object] = {"name": name}
    if tags is not None:
        cfg_kwargs["tags"] = tags
    if eval_config is not None:
        cfg_kwargs["eval_config"] = eval_config
    return ExperimentRun(
        run_id=run_id,
        config=_make_experiment_config(**cfg_kwargs),
        report=_make_report(run_id=run_id, metrics=metrics),
        status=ExperimentStatus.COMPLETED,
        started_at=started_at or datetime(2025, 1, 1, tzinfo=UTC),
        completed_at=datetime(2025, 1, 1, 0, 1, tzinfo=UTC),
        git_sha="abc123",
        duration_seconds=60.0,
    )


class _FakeRetriever:
    """Deterministic retriever for testing."""

    def __init__(self, results: dict[str, list[str]] | None = None) -> None:
        self._results = results or {}

    def retrieve(self, query: str, k: int) -> list[str]:
        return self._results.get(query, ["chunk-ml-001", "chunk-ml-002"])[:k]


# ── TestExperimentConfig ─────────────────────────────────────────────


class TestExperimentConfig:
    def test_valid_config(self) -> None:
        cfg = _make_experiment_config()
        assert cfg.name == "test-exp"
        assert cfg.k_values == [5, 10, 20]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            _make_experiment_config(name="")

    def test_empty_hypothesis_rejected(self) -> None:
        with pytest.raises(ValueError, match="hypothesis must not be empty"):
            _make_experiment_config(hypothesis="")

    def test_empty_dataset_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="dataset_path must not be empty"):
            _make_experiment_config(dataset_path="")

    def test_empty_artifact_dir_rejected(self) -> None:
        with pytest.raises(ValueError, match="artifact_dir must not be empty"):
            _make_experiment_config(artifact_dir="")


# ── TestExperimentRun ────────────────────────────────────────────────


class TestExperimentRun:
    def test_construction(self) -> None:
        run = _make_run()
        assert run.run_id == "run-1"
        assert run.status == ExperimentStatus.COMPLETED
        assert run.error is None

    def test_schema_version_default(self) -> None:
        run = _make_run()
        assert run.schema_version == 1


# ── TestInMemoryExperimentStore ──────────────────────────────────────


class TestInMemoryExperimentStore:
    def test_save_and_get(self) -> None:
        store = InMemoryExperimentStore()
        run = _make_run(run_id="r1")
        store.save(run)
        assert store.get("r1") == run

    def test_get_missing_returns_none(self) -> None:
        store = InMemoryExperimentStore()
        assert store.get("nonexistent") is None

    def test_list_all_ordered_by_started_at(self) -> None:
        store = InMemoryExperimentStore()
        early = _make_run(
            run_id="r1",
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        late = _make_run(
            run_id="r2",
            started_at=datetime(2025, 6, 1, tzinfo=UTC),
        )
        store.save(early)
        store.save(late)
        runs = store.list_all()
        assert [r.run_id for r in runs] == ["r2", "r1"]

    def test_list_by_tag(self) -> None:
        store = InMemoryExperimentStore()
        store.save(_make_run(run_id="r1", tags=["ablation", "chunk-size"]))
        store.save(_make_run(run_id="r2", tags=["ablation"]))
        store.save(_make_run(run_id="r3", tags=["production"]))
        results = store.list_by_tag("ablation")
        assert {r.run_id for r in results} == {"r1", "r2"}

    def test_list_by_name(self) -> None:
        store = InMemoryExperimentStore()
        store.save(_make_run(run_id="r1", name="exp-a"))
        store.save(_make_run(run_id="r2", name="exp-b"))
        store.save(_make_run(run_id="r3", name="exp-a"))
        results = store.list_by_name("exp-a")
        assert {r.run_id for r in results} == {"r1", "r3"}


# ── TestExperimentRunner ─────────────────────────────────────────────


class TestExperimentRunner:
    def test_success_with_artifacts(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "dataset.json"
        save_dataset(
            [
                EvaluationCase(
                    case_id="c1",
                    query="What is machine learning?",
                    expected_answer="ML is a subset of AI.",
                    relevant_chunk_ids=["chunk-ml-001"],
                ),
            ],
            ds_path,
        )
        artifact_dir = tmp_path / "artifacts"
        config = _make_experiment_config(
            dataset_path=str(ds_path),
            artifact_dir=str(artifact_dir),
        )
        retriever = _FakeRetriever()
        runner = ExperimentRunner()

        with patch("libs.experiments.runner.get_git_sha", return_value="deadbeef"):
            run = runner.run(config, retriever, run_id="test-run")

        assert run.status == ExperimentStatus.COMPLETED
        assert run.git_sha == "deadbeef"
        assert (artifact_dir / "test-run.json").exists()
        assert (artifact_dir / "test-run.md").exists()

    def test_saves_to_store(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "dataset.json"
        save_dataset(
            [
                EvaluationCase(
                    case_id="c1",
                    query="test",
                    expected_answer="answer",
                    relevant_chunk_ids=["chunk-1"],
                ),
            ],
            ds_path,
        )
        config = _make_experiment_config(
            dataset_path=str(ds_path),
            artifact_dir=str(tmp_path / "out"),
        )
        store = InMemoryExperimentStore()
        runner = ExperimentRunner(store=store)

        with patch("libs.experiments.runner.get_git_sha", return_value="abc"):
            run = runner.run(config, _FakeRetriever(), run_id="store-run")

        assert store.get("store-run") == run

    def test_captures_git_sha(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "dataset.json"
        save_dataset(
            [
                EvaluationCase(
                    case_id="c1",
                    query="q",
                    expected_answer="a",
                    relevant_chunk_ids=["c1"],
                ),
            ],
            ds_path,
        )
        config = _make_experiment_config(
            dataset_path=str(ds_path),
            artifact_dir=str(tmp_path / "out"),
        )
        with patch("libs.experiments.runner.get_git_sha", return_value="sha256hash"):
            run = ExperimentRunner().run(config, _FakeRetriever())

        assert run.git_sha == "sha256hash"

    def test_creates_artifact_dir(self, tmp_path: Path) -> None:
        ds_path = tmp_path / "dataset.json"
        save_dataset(
            [
                EvaluationCase(
                    case_id="c1",
                    query="q",
                    expected_answer="a",
                    relevant_chunk_ids=["c1"],
                ),
            ],
            ds_path,
        )
        nested_dir = tmp_path / "deep" / "nested" / "artifacts"
        config = _make_experiment_config(
            dataset_path=str(ds_path),
            artifact_dir=str(nested_dir),
        )
        with patch("libs.experiments.runner.get_git_sha", return_value="x"):
            ExperimentRunner().run(config, _FakeRetriever())

        assert nested_dir.exists()

    def test_failure_on_bad_dataset_path(self, tmp_path: Path) -> None:
        config = _make_experiment_config(
            dataset_path="/nonexistent/dataset.json",
            artifact_dir=str(tmp_path / "out"),
        )
        with patch("libs.experiments.runner.get_git_sha", return_value="x"):
            run = ExperimentRunner().run(config, _FakeRetriever())

        assert run.status == ExperimentStatus.FAILED
        assert run.error is not None


# ── TestCompareRuns ──────────────────────────────────────────────────


class TestCompareRuns:
    def test_identical_runs(self) -> None:
        m = {"recall@5": 0.8, "precision@5": 0.6, "mrr": 0.7}
        baseline = _make_run(run_id="b", metrics=m)
        candidate = _make_run(run_id="c", metrics=m)
        comparison = compare_runs(baseline, candidate)
        for d in comparison.deltas:
            assert d.absolute_change == 0.0
            assert not d.improved

    def test_improved_candidate(self) -> None:
        baseline = _make_run(
            run_id="b",
            metrics={"recall@5": 0.6, "precision@5": 0.4},
        )
        candidate = _make_run(
            run_id="c",
            metrics={"recall@5": 0.8, "precision@5": 0.6},
        )
        comparison = compare_runs(baseline, candidate)
        for d in comparison.deltas:
            assert d.improved
            assert d.absolute_change > 0

    def test_degraded_candidate(self) -> None:
        baseline = _make_run(
            run_id="b",
            metrics={"recall@5": 0.9},
        )
        candidate = _make_run(
            run_id="c",
            metrics={"recall@5": 0.5},
        )
        comparison = compare_runs(baseline, candidate)
        delta = comparison.deltas[0]
        assert not delta.improved
        assert delta.absolute_change < 0

    def test_config_diffs_detected(self) -> None:
        baseline = _make_run(
            run_id="b",
            eval_config=_make_eval_config(retrieval_mode="dense"),
        )
        candidate = _make_run(
            run_id="c",
            eval_config=_make_eval_config(retrieval_mode="hybrid"),
        )
        comparison = compare_runs(baseline, candidate)
        assert "retrieval_mode" in comparison.config_diffs
        assert comparison.config_diffs["retrieval_mode"] == ("dense", "hybrid")

    def test_markdown_format(self) -> None:
        baseline = _make_run(run_id="b", metrics={"recall@5": 0.6})
        candidate = _make_run(run_id="c", metrics={"recall@5": 0.8})
        comparison = compare_runs(baseline, candidate)
        md = generate_comparison_markdown(comparison)
        assert "# Experiment Comparison" in md
        assert "recall@5" in md
        assert "Baseline" in md
        assert "Candidate" in md


# ── TestSampleConfigs ────────────────────────────────────────────────


class TestSampleConfigs:
    def test_chunk_size_experiment(self) -> None:
        cfg = chunk_size_experiment("/data/ds.json", "/out", 256)
        assert cfg.name == "chunk-size-256"
        assert "256" in cfg.hypothesis
        assert "chunk-size" in cfg.tags
        assert cfg.eval_config.chunking_strategy == "fixed-256"

    def test_retrieval_mode_experiment(self) -> None:
        cfg = retrieval_mode_experiment("/data/ds.json", "/out", "hybrid")
        assert cfg.name == "retrieval-mode-hybrid"
        assert cfg.eval_config.retrieval_mode == "hybrid"
        assert "retrieval-mode" in cfg.tags

    def test_reranker_experiment(self) -> None:
        cfg = reranker_experiment("/data/ds.json", "/out", "cross-encoder-v1")
        assert cfg.name == "reranker-cross-encoder-v1"
        assert cfg.eval_config.reranker_id == "cross-encoder-v1"
        assert "reranker" in cfg.tags


# ── TestIntegration ──────────────────────────────────────────────────


class TestIntegration:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """Create dataset → run two experiments → compare."""
        ds_path = tmp_path / "dataset.json"
        save_dataset(
            [
                EvaluationCase(
                    case_id="c1",
                    query="What is machine learning?",
                    expected_answer="ML is AI.",
                    relevant_chunk_ids=["chunk-ml-001", "chunk-ml-002"],
                ),
                EvaluationCase(
                    case_id="c2",
                    query="What is a vector database?",
                    expected_answer="Stores vectors.",
                    relevant_chunk_ids=["chunk-vdb-001"],
                ),
            ],
            ds_path,
        )

        # Good retriever (returns relevant chunks)
        good = _FakeRetriever({
            "What is machine learning?": ["chunk-ml-001", "chunk-ml-002", "x"],
            "What is a vector database?": ["chunk-vdb-001", "y", "z"],
        })

        # Bad retriever (returns irrelevant chunks)
        bad = _FakeRetriever({
            "What is machine learning?": ["x", "y", "z"],
            "What is a vector database?": ["a", "b", "c"],
        })

        store = InMemoryExperimentStore()
        runner = ExperimentRunner(store=store)

        config_a = _make_experiment_config(
            name="good-retriever",
            dataset_path=str(ds_path),
            artifact_dir=str(tmp_path / "good"),
        )
        config_b = _make_experiment_config(
            name="bad-retriever",
            dataset_path=str(ds_path),
            artifact_dir=str(tmp_path / "bad"),
        )

        with patch("libs.experiments.runner.get_git_sha", return_value="abc"):
            run_a = runner.run(config_a, good, run_id="run-good")
            run_b = runner.run(config_b, bad, run_id="run-bad")

        assert run_a.status == ExperimentStatus.COMPLETED
        assert run_b.status == ExperimentStatus.COMPLETED
        assert len(store.list_all()) == 2

        # Compare: good should beat bad
        comparison = compare_runs(run_b, run_a)
        recall_delta = next(
            d for d in comparison.deltas if d.metric_name == "retrieval.recall@5"
        )
        assert recall_delta.improved
