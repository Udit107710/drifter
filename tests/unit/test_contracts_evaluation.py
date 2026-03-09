"""Tests for evaluation contracts: EvaluationCase, EvaluationResult."""

from datetime import UTC, datetime

import pytest

from libs.contracts.evaluation import EvaluationCase, EvaluationResult

# ── EvaluationCase ──────────────────────────────────────────────────


class TestEvaluationCase:
    def test_create_valid(self) -> None:
        case = EvaluationCase(
            case_id="eval-001",
            query="What is RAG?",
            expected_answer="Retrieval Augmented Generation.",
            relevant_chunk_ids=["chk-001", "chk-002"],
        )
        assert case.case_id == "eval-001"
        assert len(case.relevant_chunk_ids) == 2
        assert case.metadata == {}

    def test_schema_version_default(self) -> None:
        case = EvaluationCase(
            case_id="eval-001",
            query="What is RAG?",
            expected_answer="Retrieval Augmented Generation.",
            relevant_chunk_ids=["chk-001"],
        )
        assert case.schema_version == 1

    def test_schema_version_custom(self) -> None:
        case = EvaluationCase(
            case_id="eval-001",
            query="What is RAG?",
            expected_answer="Retrieval Augmented Generation.",
            relevant_chunk_ids=["chk-001"],
            schema_version=2,
        )
        assert case.schema_version == 2

    def test_with_metadata(self) -> None:
        case = EvaluationCase(
            case_id="eval-001",
            query="q",
            expected_answer="a",
            relevant_chunk_ids=["chk-001"],
            metadata={"difficulty": "easy"},
        )
        assert case.metadata["difficulty"] == "easy"

    def test_empty_case_id_raises(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            EvaluationCase(
                case_id="",
                query="q",
                expected_answer="a",
                relevant_chunk_ids=["chk-001"],
            )

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            EvaluationCase(
                case_id="eval-001",
                query="",
                expected_answer="a",
                relevant_chunk_ids=["chk-001"],
            )

    def test_empty_expected_answer_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_answer"):
            EvaluationCase(
                case_id="eval-001",
                query="q",
                expected_answer="",
                relevant_chunk_ids=["chk-001"],
            )

    def test_empty_relevant_chunk_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="relevant_chunk_ids"):
            EvaluationCase(
                case_id="eval-001",
                query="q",
                expected_answer="a",
                relevant_chunk_ids=[],
            )


# ── EvaluationResult ────────────────────────────────────────────────


class TestEvaluationResult:
    def test_create_valid(self) -> None:
        result = EvaluationResult(
            run_id="run-001",
            case_id="eval-001",
            metrics={"recall_at_5": 0.8, "mrr": 0.75},
            config={"chunking": "fixed_size", "embedding_model": "bge-base"},
            evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert result.metrics["recall_at_5"] == 0.8
        assert result.metadata == {}

    def test_schema_version_default(self) -> None:
        result = EvaluationResult(
            run_id="run-001",
            case_id="eval-001",
            metrics={"recall_at_5": 0.8},
            config={},
            evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert result.schema_version == 1

    def test_schema_version_custom(self) -> None:
        result = EvaluationResult(
            run_id="run-001",
            case_id="eval-001",
            metrics={"recall_at_5": 0.8},
            config={},
            evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
            schema_version=2,
        )
        assert result.schema_version == 2

    def test_empty_run_id_raises(self) -> None:
        with pytest.raises(ValueError, match="run_id"):
            EvaluationResult(
                run_id="",
                case_id="eval-001",
                metrics={"recall_at_5": 0.8},
                config={},
                evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_empty_case_id_raises(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            EvaluationResult(
                run_id="run-001",
                case_id="",
                metrics={"recall_at_5": 0.8},
                config={},
                evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="metrics"):
            EvaluationResult(
                run_id="run-001",
                case_id="eval-001",
                metrics={},
                config={},
                evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
