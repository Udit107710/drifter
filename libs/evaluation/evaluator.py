"""Evaluator: runs evaluation cases and computes retrieval + answer metrics."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from libs.contracts.evaluation import EvaluationCase
from libs.evaluation.models import (
    EvaluationConfig,
    EvaluationReport,
    QueryResult,
    StageMetrics,
)
from libs.evaluation.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@runtime_checkable
class Retriever(Protocol):
    """Protocol for a retriever that can be evaluated."""
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return top-k chunk IDs for a query."""
        ...


class RetrievalEvaluator:
    """Evaluates retrieval quality against ground-truth cases.

    Computes Recall@k, Precision@k, MRR, and NDCG@k for each case,
    then aggregates into means and medians.
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        run_id: str = "",
    ) -> None:
        self._config = config or EvaluationConfig()
        self._run_id = run_id or f"eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    def evaluate(
        self,
        cases: list[EvaluationCase],
        retriever: Retriever,
        k_values: list[int] | None = None,
        relevance_grades: dict[str, dict[str, int]] | None = None,
    ) -> EvaluationReport:
        """Run evaluation over all cases.

        Args:
            cases: Ground-truth evaluation cases.
            retriever: Implementation that returns chunk IDs for queries.
            k_values: List of k values to compute metrics at. Defaults to config.
            relevance_grades: Optional per-case graded relevance for NDCG.
                Maps case_id -> {chunk_id: grade}.
        """
        ks = k_values or self._config.k_values or [5, 10, 20]
        query_results: list[QueryResult] = []

        for case in cases:
            max_k = max(ks)
            retrieved = retriever.retrieve(case.query, max_k)
            relevant = set(case.relevant_chunk_ids)

            metrics: dict[str, float] = {}
            for k in ks:
                metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
                metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            metrics["mrr"] = mrr(retrieved, relevant)

            # NDCG if graded relevance available
            case_grades = (relevance_grades or {}).get(case.case_id)
            if case_grades:
                for k in ks:
                    metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, case_grades, k)
            else:
                # Binary relevance: relevant=1, not relevant=0
                binary_grades = {cid: 1 for cid in case.relevant_chunk_ids}
                for k in ks:
                    metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, binary_grades, k)

            query_results.append(QueryResult(
                case_id=case.case_id,
                query=case.query,
                retrieved_ids=retrieved[:max_k],
                relevant_ids=case.relevant_chunk_ids,
                metrics=metrics,
                metadata=case.metadata,
            ))

        # Aggregate into stage metrics
        stage_metrics = self._aggregate(query_results)

        return EvaluationReport(
            run_id=self._run_id,
            config=self._config,
            query_results=query_results,
            stage_metrics=stage_metrics,
            evaluated_at=datetime.now(UTC),
            dataset_name="",
            dataset_size=len(cases),
        )

    def _aggregate(self, results: list[QueryResult]) -> list[StageMetrics]:
        """Compute mean and median for each metric across all queries."""
        if not results:
            return []

        # Collect all metric names
        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.metrics.keys())

        means: dict[str, float] = {}
        medians: dict[str, float] = {}
        for key in sorted(all_keys):
            values = [r.metrics[key] for r in results if key in r.metrics]
            if values:
                means[key] = statistics.mean(values)
                medians[key] = statistics.median(values)

        return [StageMetrics(
            stage="retrieval",
            metric_means=means,
            metric_medians=medians,
            query_count=len(results),
        )]
