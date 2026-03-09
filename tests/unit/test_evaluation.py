"""Comprehensive tests for the evaluation subsystem."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from libs.contracts.evaluation import EvaluationCase
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.evaluation.answer_metrics import (
    detect_unsupported_claims,
    evaluate_citation_accuracy,
    evaluate_faithfulness,
)
from libs.evaluation.dataset import create_seed_dataset, load_dataset, save_dataset
from libs.evaluation.evaluator import RetrievalEvaluator
from libs.evaluation.models import (
    EvaluationConfig,
    EvaluationReport,
    QueryResult,
    StageMetrics,
)
from libs.evaluation.report import (
    generate_markdown_summary,
    report_to_dict,
    save_json_report,
)
from libs.evaluation.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def _citation(claim: str, chunk_id: str) -> Citation:
    return Citation(
        claim=claim,
        chunk_id=chunk_id,
        chunk_content="some content",
        source_id="src-1",
        confidence=0.9,
    )


def _generated_answer(citations: list[Citation]) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer="Test answer",
        citations=citations,
        model_id="test-model",
        token_usage=_token_usage(),
        trace_id="trace-1",
    )


class MockRetriever:
    """Retriever that returns a fixed list of IDs regardless of query."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def retrieve(self, query: str, k: int) -> list[str]:
        return self._ids[:k]


class PerfectRetriever:
    """Retriever that always returns the relevant chunk IDs for known cases."""

    def __init__(self, case_map: dict[str, list[str]]) -> None:
        self._map = case_map

    def retrieve(self, query: str, k: int) -> list[str]:
        return self._map.get(query, [])[:k]


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "e", "f"}
        # top-3: {a, b, c} & {a, c, e, f} = {a, c} -> 2/4 = 0.5
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_zero_recall(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_empty_relevant(self) -> None:
        retrieved = ["a", "b"]
        assert recall_at_k(retrieved, set(), 2) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        # top-10 of 2 items: {a, b} & {a, b, c} = {a, b} -> 2/3
        assert recall_at_k(retrieved, relevant, 10) == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_precision(self) -> None:
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        # top-4: 2 relevant / 4 = 0.5
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_zero_precision(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_k_zero(self) -> None:
        retrieved = ["a", "b"]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, 0) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_is_relevant(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self) -> None:
        assert mrr(["x", "a", "b"], {"a"}) == 0.5

    def test_none_relevant(self) -> None:
        assert mrr(["x", "y", "z"], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# NDCG@k
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        # Items already in ideal order
        retrieved = ["a", "b", "c"]
        grades = {"a": 3, "b": 2, "c": 1}
        assert ndcg_at_k(retrieved, grades, 3) == pytest.approx(1.0)

    def test_reversed_ranking(self) -> None:
        retrieved = ["c", "b", "a"]
        grades = {"a": 3, "b": 2, "c": 1}
        score = ndcg_at_k(retrieved, grades, 3)
        assert 0.0 < score < 1.0

    def test_binary_relevance(self) -> None:
        retrieved = ["a", "x", "b"]
        grades = {"a": 1, "b": 1}
        score = ndcg_at_k(retrieved, grades, 3)
        assert 0.0 < score <= 1.0

    def test_graded_relevance(self) -> None:
        # Verify graded relevance produces different scores for different orderings
        grades = {"a": 3, "b": 2, "c": 1}
        perfect = ndcg_at_k(["a", "b", "c"], grades, 3)
        worse = ndcg_at_k(["c", "b", "a"], grades, 3)
        assert perfect > worse

    def test_empty_grades(self) -> None:
        assert ndcg_at_k(["a", "b"], {}, 2) == 0.0

    def test_k_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a": 1}, 0) == 0.0


# ---------------------------------------------------------------------------
# Citation Accuracy
# ---------------------------------------------------------------------------


class TestCitationAccuracy:
    def test_all_valid(self) -> None:
        answer = _generated_answer([
            _citation("claim1", "c1"),
            _citation("claim2", "c2"),
        ])
        result = evaluate_citation_accuracy(answer, {"c1", "c2", "c3"})
        assert result.score == 1.0
        assert result.total_citations == 2
        assert result.valid_citations == 2
        assert result.invalid_citations == []

    def test_some_invalid(self) -> None:
        answer = _generated_answer([
            _citation("claim1", "c1"),
            _citation("claim2", "c_bad"),
        ])
        result = evaluate_citation_accuracy(answer, {"c1", "c2"})
        assert result.score == 0.5
        assert result.valid_citations == 1
        assert result.invalid_citations == ["c_bad"]

    def test_no_citations(self) -> None:
        answer = _generated_answer([])
        result = evaluate_citation_accuracy(answer, {"c1"})
        assert result.score == 1.0
        assert result.total_citations == 0

    def test_all_invalid(self) -> None:
        answer = _generated_answer([
            _citation("claim1", "bad1"),
            _citation("claim2", "bad2"),
        ])
        result = evaluate_citation_accuracy(answer, {"c1", "c2"})
        assert result.score == 0.0
        assert result.valid_citations == 0
        assert result.invalid_citations == ["bad1", "bad2"]


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def test_all_supported(self) -> None:
        claims = ["claim_a", "claim_b"]
        supported = {"c1", "c2"}
        claim_to_chunk = {"claim_a": "c1", "claim_b": "c2"}
        result = evaluate_faithfulness(claims, supported, claim_to_chunk)
        assert result.score == 1.0
        assert result.total_claims == 2
        assert result.supported_claims == 2
        assert result.unsupported_claims == []

    def test_some_unsupported(self) -> None:
        claims = ["claim_a", "claim_b"]
        supported = {"c1"}
        claim_to_chunk = {"claim_a": "c1", "claim_b": "c_missing"}
        result = evaluate_faithfulness(claims, supported, claim_to_chunk)
        assert result.score == 0.5
        assert result.supported_claims == 1
        assert result.unsupported_claims == ["claim_b"]

    def test_no_claims(self) -> None:
        result = evaluate_faithfulness([], set(), {})
        assert result.score == 1.0
        assert result.total_claims == 0

    def test_unmapped_claim(self) -> None:
        claims = ["claim_a", "claim_b"]
        supported = {"c1"}
        claim_to_chunk = {"claim_a": "c1"}  # claim_b not mapped
        result = evaluate_faithfulness(claims, supported, claim_to_chunk)
        assert result.score == 0.5
        assert "claim_b" in result.unsupported_claims


# ---------------------------------------------------------------------------
# Unsupported Claims Detection
# ---------------------------------------------------------------------------


class TestUnsupportedClaims:
    def test_no_unsupported(self) -> None:
        answer = _generated_answer([
            _citation("claim1", "c1"),
            _citation("claim2", "c2"),
        ])
        result = detect_unsupported_claims(answer, {"c1", "c2"})
        assert result == []

    def test_some_unsupported(self) -> None:
        answer = _generated_answer([
            _citation("claim1", "c1"),
            _citation("claim2", "c_bad"),
        ])
        result = detect_unsupported_claims(answer, {"c1"})
        assert result == ["claim2"]


# ---------------------------------------------------------------------------
# Seed Dataset
# ---------------------------------------------------------------------------


class TestSeedDataset:
    def test_create_seed(self) -> None:
        cases = create_seed_dataset()
        assert len(cases) == 5
        for case in cases:
            assert case.case_id
            assert case.query
            assert case.expected_answer
            assert len(case.relevant_chunk_ids) > 0
            assert isinstance(case, EvaluationCase)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        cases = create_seed_dataset()
        file_path = tmp_path / "dataset.json"
        save_dataset(cases, file_path)
        loaded = load_dataset(file_path)
        assert len(loaded) == len(cases)
        for original, restored in zip(cases, loaded, strict=False):
            assert original.case_id == restored.case_id
            assert original.query == restored.query
            assert original.expected_answer == restored.expected_answer
            assert original.relevant_chunk_ids == restored.relevant_chunk_ids


# ---------------------------------------------------------------------------
# RetrievalEvaluator
# ---------------------------------------------------------------------------


class TestRetrievalEvaluator:
    def _seed_cases(self) -> list[EvaluationCase]:
        return [
            EvaluationCase(
                case_id="t1",
                query="query one",
                expected_answer="answer one",
                relevant_chunk_ids=["a", "b"],
            ),
            EvaluationCase(
                case_id="t2",
                query="query two",
                expected_answer="answer two",
                relevant_chunk_ids=["c", "d"],
            ),
        ]

    def test_evaluate_basic(self) -> None:
        cases = self._seed_cases()
        retriever = MockRetriever(["a", "x", "y", "z", "b"])
        evaluator = RetrievalEvaluator(run_id="test-run")
        report = evaluator.evaluate(cases, retriever, k_values=[5])
        assert len(report.query_results) == 2
        assert len(report.stage_metrics) == 1
        assert report.stage_metrics[0].stage == "retrieval"
        # Ensure metrics keys exist
        qr = report.query_results[0]
        assert "recall@5" in qr.metrics
        assert "precision@5" in qr.metrics
        assert "mrr" in qr.metrics
        assert "ndcg@5" in qr.metrics

    def test_evaluate_perfect_retriever(self) -> None:
        cases = self._seed_cases()
        case_map = {c.query: c.relevant_chunk_ids for c in cases}
        retriever = PerfectRetriever(case_map)
        evaluator = RetrievalEvaluator(run_id="perfect-run")
        report = evaluator.evaluate(cases, retriever, k_values=[5])
        for qr in report.query_results:
            assert qr.metrics["recall@5"] == pytest.approx(1.0)
            assert qr.metrics["mrr"] == pytest.approx(1.0)
            assert qr.metrics["ndcg@5"] == pytest.approx(1.0)

    def test_evaluate_empty_retriever(self) -> None:
        cases = self._seed_cases()
        retriever = MockRetriever([])
        evaluator = RetrievalEvaluator(run_id="empty-run")
        report = evaluator.evaluate(cases, retriever, k_values=[5])
        for qr in report.query_results:
            assert qr.metrics["recall@5"] == 0.0
            assert qr.metrics["precision@5"] == 0.0
            assert qr.metrics["mrr"] == 0.0
            assert qr.metrics["ndcg@5"] == 0.0

    def test_custom_k_values(self) -> None:
        cases = self._seed_cases()
        retriever = MockRetriever(["a", "b", "c", "d"])
        evaluator = RetrievalEvaluator(run_id="custom-k")
        report = evaluator.evaluate(cases, retriever, k_values=[1, 3])
        qr = report.query_results[0]
        assert "recall@1" in qr.metrics
        assert "recall@3" in qr.metrics
        assert "precision@1" in qr.metrics
        assert "precision@3" in qr.metrics

    def test_graded_relevance_ndcg(self) -> None:
        cases = [
            EvaluationCase(
                case_id="g1",
                query="graded query",
                expected_answer="graded answer",
                relevant_chunk_ids=["a", "b", "c"],
            ),
        ]
        retriever = MockRetriever(["a", "b", "c"])
        grades = {"g1": {"a": 3, "b": 2, "c": 1}}
        evaluator = RetrievalEvaluator(run_id="graded-run")
        report = evaluator.evaluate(cases, retriever, k_values=[3], relevance_grades=grades)
        qr = report.query_results[0]
        # Perfect order with graded relevance -> NDCG = 1.0
        assert qr.metrics["ndcg@3"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def _make_report(self) -> EvaluationReport:
        from datetime import UTC, datetime

        return EvaluationReport(
            run_id="rpt-001",
            config=EvaluationConfig(
                retrieval_mode="dense",
                embedding_model="test-embed",
                k_values=[5, 10],
            ),
            query_results=[
                QueryResult(
                    case_id="q1",
                    query="test query",
                    retrieved_ids=["a", "b"],
                    relevant_ids=["a"],
                    metrics={"recall@5": 1.0, "precision@5": 0.5, "mrr": 1.0, "ndcg@5": 1.0},
                ),
            ],
            stage_metrics=[
                StageMetrics(
                    stage="retrieval",
                    metric_means={"recall@5": 1.0, "precision@5": 0.5, "mrr": 1.0, "ndcg@5": 1.0},
                    metric_medians={"recall@5": 1.0, "precision@5": 0.5, "mrr": 1.0, "ndcg@5": 1.0},
                    query_count=1,
                ),
            ],
            evaluated_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            dataset_name="test-dataset",
            dataset_size=1,
        )

    def test_report_to_dict(self) -> None:
        report = self._make_report()
        d = report_to_dict(report)
        assert "run_id" in d
        assert "config" in d
        assert "query_results" in d
        assert "stage_metrics" in d
        assert "evaluated_at" in d
        assert "dataset_name" in d
        assert "dataset_size" in d
        assert isinstance(d["evaluated_at"], str)

    def test_markdown_summary(self) -> None:
        report = self._make_report()
        md = generate_markdown_summary(report)
        assert "Evaluation Report" in md
        assert "rpt-001" in md
        assert "Configuration" in md
        assert "Aggregate Metrics" in md
        assert "Per-Query Results" in md
        assert "recall@5" in md
        assert "dense" in md

    def test_save_json_roundtrip(self, tmp_path: Path) -> None:
        report = self._make_report()
        json_path = tmp_path / "report.json"
        save_json_report(report, json_path)
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["run_id"] == "rpt-001"
        assert data["dataset_size"] == 1
        assert len(data["query_results"]) == 1
        assert data["query_results"][0]["metrics"]["recall@5"] == 1.0

    def test_aggregate_metrics_property(self) -> None:
        report = self._make_report()
        agg = report.aggregate_metrics
        assert "retrieval.recall@5" in agg
        assert "retrieval.precision@5" in agg
        assert "retrieval.mrr" in agg
        assert agg["retrieval.recall@5"] == 1.0


# ---------------------------------------------------------------------------
# Integration: seed -> evaluate -> report
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_seed_to_report(self, tmp_path: Path) -> None:
        # 1. Create seed dataset
        cases = create_seed_dataset()
        assert len(cases) == 5

        # 2. Build a mock retriever that returns the first relevant chunk for each case
        case_map = {c.query: c.relevant_chunk_ids for c in cases}
        retriever = PerfectRetriever(case_map)

        # 3. Run evaluation
        config = EvaluationConfig(
            retrieval_mode="mock",
            embedding_model="none",
            k_values=[5],
        )
        evaluator = RetrievalEvaluator(config=config, run_id="integration-test")
        report = evaluator.evaluate(cases, retriever, k_values=[5])

        # 4. Verify report structure
        assert report.run_id == "integration-test"
        assert report.dataset_size == 5
        assert len(report.query_results) == 5
        assert len(report.stage_metrics) == 1

        # 5. All metrics should be perfect since PerfectRetriever returns relevant IDs
        for qr in report.query_results:
            assert qr.metrics["recall@5"] == pytest.approx(1.0)
            assert qr.metrics["mrr"] == pytest.approx(1.0)

        # 6. Generate and save reports
        md = generate_markdown_summary(report)
        assert "integration-test" in md

        json_path = tmp_path / "integration_report.json"
        save_json_report(report, json_path)
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["run_id"] == "integration-test"
        assert len(data["query_results"]) == 5

        # 7. Aggregate metrics accessible
        agg = report.aggregate_metrics
        assert "retrieval.recall@5" in agg
        assert agg["retrieval.recall@5"] == pytest.approx(1.0)
