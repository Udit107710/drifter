"""Output rendering for CLI: human-readable and JSON modes."""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from datetime import datetime
from enum import Enum
from typing import Any

from libs.context_builder.models import BuilderResult
from libs.evaluation.models import EvaluationReport
from libs.generation.models import GenerationResult
from libs.reranking.models import RerankerResult
from libs.retrieval.broker.models import BrokerResult
from orchestrators.query import QueryResult


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, datetimes to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _serialize(getattr(obj, f.name)) for f in fields(obj)}
    return str(obj)


class OutputRenderer:
    """Renders pipeline results to stdout/stderr."""

    def __init__(self, json_mode: bool = False, verbose: bool = False) -> None:
        self._json_mode = json_mode
        self._verbose = verbose

    def render_query_result(self, result: QueryResult) -> None:
        """Render a full query pipeline result."""
        if self._json_mode:
            self._emit_json({
                "trace_id": result.trace_id,
                "outcome": result.outcome,
                "latency_ms": round(result.total_latency_ms, 2),
                "query": result.query,
                "data": {
                    "broker": _serialize(result.broker_result),
                    "reranker": _serialize(result.reranker_result),
                    "builder": _serialize(result.builder_result),
                    "generation": _serialize(result.generation_result),
                },
                "errors": result.errors,
            })
            return

        self._meta(f"trace: {result.trace_id}")
        self._meta(f"outcome: {result.outcome}")
        self._meta(f"latency: {result.total_latency_ms:.1f}ms")

        if result.errors:
            for e in result.errors:
                self._meta(f"warning: {e}")

        # Display answer if available
        if result.generation_result and result.generation_result.answer:
            answer = result.generation_result.answer
            print(f"\n{answer.answer}")

            if answer.citations:
                print("\nCitations:")
                for i, c in enumerate(answer.citations, 1):
                    print(f"  [{i}] {c.claim}")
                    print(f"      source: {c.source_id}, chunk: {c.chunk_id}")
                    print(f"      confidence: {c.confidence:.2f}")
        elif result.outcome == "no_results":
            print("\nNo results found.")

        if self._verbose:
            self._render_verbose_query(result)

    def render_broker_result(self, result: BrokerResult, trace_id: str) -> None:
        """Render retrieval-only results."""
        if self._json_mode:
            self._emit_json({
                "trace_id": trace_id,
                "outcome": result.outcome.value,
                "latency_ms": round(result.total_latency_ms, 2),
                "data": _serialize(result),
                "errors": result.errors,
            })
            return

        self._meta(f"trace: {trace_id}")
        self._meta(f"outcome: {result.outcome.value}")
        self._meta(f"mode: {result.mode.value}")
        self._meta(f"candidates: {result.candidate_count}")
        self._meta(f"latency: {result.total_latency_ms:.1f}ms")

        for i, c in enumerate(result.candidates, 1):
            print(f"\n  [{i}] score={c.fused_score:.4f} method={c.retrieval_method.value}")
            print(f"      chunk={c.chunk.chunk_id}")
            content_preview = c.chunk.content[:120].replace("\n", " ")
            print(f"      {content_preview}")

    def render_reranker_result(self, result: RerankerResult, trace_id: str) -> None:
        """Render reranking results."""
        if self._json_mode:
            self._emit_json({
                "trace_id": trace_id,
                "outcome": result.outcome.value,
                "latency_ms": round(result.total_latency_ms, 2),
                "data": _serialize(result),
                "errors": result.errors,
            })
            return

        self._meta(f"trace: {trace_id}")
        self._meta(f"outcome: {result.outcome.value}")
        self._meta(f"reranker: {result.reranker_id}")
        self._meta(f"candidates: {result.candidate_count}")
        self._meta(f"latency: {result.total_latency_ms:.1f}ms")

        for rc in result.ranked_candidates:
            print(
                f"\n  [{rc.rank}] rerank_score={rc.rerank_score:.4f} "
                f"retrieval_score={rc.candidate.score:.4f}"
            )
            print(f"      chunk={rc.candidate.chunk.chunk_id}")
            content_preview = rc.candidate.chunk.content[:120].replace("\n", " ")
            print(f"      {content_preview}")

    def render_builder_result(self, result: BuilderResult, trace_id: str) -> None:
        """Render context building results."""
        if self._json_mode:
            self._emit_json({
                "trace_id": trace_id,
                "outcome": result.outcome.value,
                "latency_ms": round(result.total_latency_ms, 2),
                "data": _serialize(result),
                "errors": result.errors,
            })
            return

        pack = result.context_pack
        self._meta(f"trace: {trace_id}")
        self._meta(f"outcome: {result.outcome.value}")
        self._meta(
            f"tokens: {pack.total_tokens}/{pack.token_budget} "
            f"({pack.total_tokens / max(pack.token_budget, 1) * 100:.0f}% used)"
        )
        self._meta(f"diversity: {pack.diversity_score:.2f}")
        self._meta(f"evidence items: {len(pack.evidence)}")
        self._meta(f"latency: {result.total_latency_ms:.1f}ms")

        for item in pack.evidence:
            print(
                f"\n  [{item.rank}] tokens={item.token_count} "
                f"reason={item.selection_reason.value}"
            )
            print(f"      chunk={item.chunk.chunk_id}")
            content_preview = item.chunk.content[:120].replace("\n", " ")
            print(f"      {content_preview}")

        if result.exclusions and self._verbose:
            print("\nExclusions:")
            for ex in result.exclusions:
                print(f"  chunk={ex.chunk_id} reason={ex.reason} tokens={ex.token_count}")

    def render_generation_result(self, result: GenerationResult, trace_id: str) -> None:
        """Render generation results."""
        if self._json_mode:
            self._emit_json({
                "trace_id": trace_id,
                "outcome": result.outcome.value,
                "latency_ms": round(result.total_latency_ms, 2),
                "data": _serialize(result),
                "errors": result.errors,
            })
            return

        self._meta(f"trace: {trace_id}")
        self._meta(f"outcome: {result.outcome.value}")
        self._meta(f"generator: {result.generator_id}")
        self._meta(f"latency: {result.total_latency_ms:.1f}ms")

        if result.answer:
            print(f"\n{result.answer.answer}")
            if result.answer.citations:
                print("\nCitations:")
                for i, c in enumerate(result.answer.citations, 1):
                    print(f"  [{i}] {c.claim}")

    def render_evaluation_report(self, report: EvaluationReport) -> None:
        """Render evaluation metrics."""
        if self._json_mode:
            self._emit_json({
                "trace_id": report.run_id,
                "outcome": "success",
                "data": _serialize(report),
                "errors": [],
            })
            return

        self._meta(f"run: {report.run_id}")
        self._meta(f"dataset_size: {report.dataset_size}")
        self._meta(f"evaluated_at: {report.evaluated_at.isoformat()}")

        if report.stage_metrics:
            print("\nMetrics:")
            for sm in report.stage_metrics:
                print(f"\n  Stage: {sm.stage} ({sm.query_count} queries)")
                for name in sorted(sm.metric_means):
                    mean = sm.metric_means[name]
                    median = sm.metric_medians.get(name, 0.0)
                    print(f"    {name:20s}  mean={mean:.4f}  median={median:.4f}")

    def render_error(self, message: str, trace_id: str | None = None) -> None:
        """Render an error message."""
        if self._json_mode:
            self._emit_json({
                "trace_id": trace_id or "",
                "outcome": "error",
                "data": None,
                "errors": [message],
            })
            return

        if trace_id:
            self._meta(f"trace: {trace_id}")
        print(f"Error: {message}", file=sys.stderr)

    def _render_verbose_query(self, result: QueryResult) -> None:
        """Print verbose stage details for a query result."""
        if result.broker_result:
            br = result.broker_result
            print("\n--- Retrieval ---")
            print(f"  mode: {br.mode.value}")
            print(f"  candidates: {br.candidate_count}")
            print(f"  stores: {len(br.store_results)}")
            for sr in br.store_results:
                print(f"    {sr.store_id}: {sr.candidate_count} candidates, {sr.latency_ms:.1f}ms")

        if result.reranker_result:
            rr = result.reranker_result
            print("\n--- Reranking ---")
            print(f"  reranker: {rr.reranker_id}")
            print(f"  candidates: {rr.candidate_count}")

        if result.builder_result:
            bld = result.builder_result
            pack = bld.context_pack
            print("\n--- Context ---")
            print(f"  tokens: {pack.total_tokens}/{pack.token_budget}")
            print(f"  evidence: {len(pack.evidence)}")
            print(f"  exclusions: {len(bld.exclusions)}")
            print(f"  diversity: {pack.diversity_score:.2f}")

    def _emit_json(self, data: dict[str, Any]) -> None:
        """Write a JSON envelope to stdout."""
        print(json.dumps(data, default=str, indent=2))

    def _meta(self, text: str) -> None:
        """Write metadata to stderr."""
        print(text, file=sys.stderr)
