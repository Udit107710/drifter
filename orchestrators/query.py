"""Query orchestrator: composes retrieval → reranking → context → generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from libs.context_builder.models import BuilderResult
from libs.context_builder.service import ContextBuilderService
from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery
from libs.generation.models import GenerationResult
from libs.generation.service import GenerationService
from libs.observability.context import ObservabilityContext
from libs.observability.stage_instruments import pipeline_span, record_stage_result
from libs.observability.tracer import Tracer
from libs.reranking.converters import fused_list_to_retrieval_candidates
from libs.reranking.models import RerankerResult
from libs.reranking.service import RerankerService
from libs.retrieval.broker.models import BrokerOutcome, BrokerResult
from libs.retrieval.broker.service import RetrievalBroker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    """Full pipeline result with all intermediate outputs."""

    trace_id: str
    query: str
    broker_result: BrokerResult | None = None
    reranker_result: RerankerResult | None = None
    builder_result: BuilderResult | None = None
    generation_result: GenerationResult | None = None
    total_latency_ms: float = 0.0
    outcome: str = "success"
    errors: list[str] = field(default_factory=list)


class QueryOrchestrator:
    """Runs the full query pipeline with observability and degraded-mode fallbacks."""

    def __init__(
        self,
        tracer: Tracer,
        retrieval_broker: RetrievalBroker,
        reranker_service: RerankerService,
        context_builder_service: ContextBuilderService,
        generation_service: GenerationService,
        token_budget: int = 3000,
    ) -> None:
        self._tracer = tracer
        self._broker = retrieval_broker
        self._reranker = reranker_service
        self._context_builder = context_builder_service
        self._generator = generation_service
        self._token_budget = token_budget

    def run(
        self,
        query: str,
        trace_id: str | None = None,
        top_k: int = 50,
        token_budget: int | None = None,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> QueryResult:
        """Execute the full query pipeline.

        Stages: retrieval → reranking → context building → generation.
        Each stage is wrapped in a pipeline_span for observability.
        If reranking fails, falls back to retrieval-order candidates.

        Args:
            on_token: Optional callback ``(text, is_thinking)`` for streaming
                      generation output to the caller in real time.
        """
        start = time.monotonic()
        ctx = self._tracer.create_context(trace_id=trace_id)
        budget = token_budget or self._token_budget

        logger.info(
            "pipeline: entry trace_id=%s query=%r top_k=%d",
            ctx.trace_id, query[:100], top_k,
        )

        with pipeline_span(self._tracer, ctx, "rag-pipeline") as root:
            root.set_attribute("query", query)
            result = self._run_full_pipeline(ctx, query, top_k, budget, start, on_token)
            record_stage_result(
                root, outcome=result.outcome,
                input_count=1, output_count=1 if result.outcome == "success" else 0,
            )
            logger.info(
                "pipeline: exit trace_id=%s outcome=%s latency=%.1fms",
                ctx.trace_id, result.outcome, result.total_latency_ms,
            )
            return result

    def _run_full_pipeline(
        self,
        ctx: ObservabilityContext,
        query: str,
        top_k: int,
        budget: int,
        start: float,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> QueryResult:
        """Inner pipeline logic, called within the root span."""
        errors: list[str] = []

        retrieval_query = RetrievalQuery(
            raw_query=query,
            normalized_query=query,
            trace_id=ctx.trace_id,
            top_k=top_k,
        )

        # --- Stage 1: Retrieval ---
        logger.info("pipeline: stage=retrieval trace_id=%s", ctx.trace_id)
        broker_result = self._run_retrieval(ctx, retrieval_query)
        logger.info(
            "pipeline: retrieval done candidates=%d outcome=%s",
            broker_result.candidate_count, broker_result.outcome.value,
        )
        if broker_result.outcome == BrokerOutcome.FAILED:
            return QueryResult(
                trace_id=ctx.trace_id,
                query=query,
                broker_result=broker_result,
                total_latency_ms=_elapsed_ms(start),
                outcome="failed",
                errors=["Retrieval failed: " + "; ".join(broker_result.errors)],
            )

        if broker_result.outcome == BrokerOutcome.NO_RESULTS:
            return QueryResult(
                trace_id=ctx.trace_id,
                query=query,
                broker_result=broker_result,
                total_latency_ms=_elapsed_ms(start),
                outcome="no_results",
                errors=[],
            )

        # --- Stage 2: Reranking ---
        logger.info("pipeline: stage=reranking trace_id=%s", ctx.trace_id)
        candidates = fused_list_to_retrieval_candidates(broker_result.candidates)
        reranker_result: RerankerResult | None = None
        try:
            reranker_result = self._run_reranking(ctx, candidates, retrieval_query)
            ranked = reranker_result.ranked_candidates
        except Exception as exc:
            logger.warning("pipeline: reranking exception, using retrieval order: %s", exc)
            errors.append(f"Reranking failed, using retrieval order: {exc}")
            ranked = []

        # Fallback: if reranking returned no candidates (timeout, error,
        # or exception), preserve retrieval order so the pipeline can
        # still generate an answer from the retrieved context.
        if not ranked and candidates:
            logger.warning(
                "pipeline: reranking produced no candidates, "
                "falling back to retrieval order (%d candidates)",
                len(candidates),
            )
            if not errors:
                errors.append("Reranking failed, using retrieval order")
            ranked = [
                RankedCandidate(
                    candidate=c,
                    rank=i + 1,
                    rerank_score=c.score,
                    reranker_id="fallback",
                )
                for i, c in enumerate(candidates)
            ]

        # --- Stage 3: Context Building ---
        logger.info(
            "pipeline: stage=context_build trace_id=%s ranked=%d",
            ctx.trace_id, len(ranked),
        )
        builder_result = self._run_context_build(ctx, ranked, query, budget)

        # --- Stage 4: Generation ---
        logger.info("pipeline: stage=generation trace_id=%s", ctx.trace_id)
        generation_result = self._run_generation(ctx, builder_result, ctx.trace_id, on_token)

        outcome = "success"
        if errors:
            outcome = "partial"

        return QueryResult(
            trace_id=ctx.trace_id,
            query=query,
            broker_result=broker_result,
            reranker_result=reranker_result,
            builder_result=builder_result,
            generation_result=generation_result,
            total_latency_ms=_elapsed_ms(start),
            outcome=outcome,
            errors=errors,
        )

    def run_retrieve_only(
        self,
        query: str,
        trace_id: str | None = None,
        top_k: int = 50,
    ) -> QueryResult:
        """Run retrieval stage only."""
        start = time.monotonic()
        ctx = self._tracer.create_context(trace_id=trace_id)
        with pipeline_span(self._tracer, ctx, "rag-pipeline") as root:
            root.set_attribute("query", query)
            rq = RetrievalQuery(
                raw_query=query, normalized_query=query,
                trace_id=ctx.trace_id, top_k=top_k,
            )
            broker_result = self._run_retrieval(ctx, rq)
            if broker_result.outcome == BrokerOutcome.SUCCESS:
                outcome = "success"
            else:
                outcome = broker_result.outcome.value
            record_stage_result(
                root, outcome=outcome,
                input_count=1, output_count=broker_result.candidate_count,
            )
            return QueryResult(
                trace_id=ctx.trace_id, query=query, broker_result=broker_result,
                total_latency_ms=_elapsed_ms(start), outcome=outcome,
            )

    def run_through_rerank(
        self,
        query: str,
        trace_id: str | None = None,
        top_k: int = 50,
    ) -> QueryResult:
        """Run retrieval + reranking."""
        start = time.monotonic()
        ctx = self._tracer.create_context(trace_id=trace_id)
        with pipeline_span(self._tracer, ctx, "rag-pipeline") as root:
            root.set_attribute("query", query)
            rq = RetrievalQuery(
                raw_query=query, normalized_query=query,
                trace_id=ctx.trace_id, top_k=top_k,
            )
            broker_result = self._run_retrieval(ctx, rq)
            if broker_result.outcome in (BrokerOutcome.FAILED, BrokerOutcome.NO_RESULTS):
                record_stage_result(
                    root, outcome=broker_result.outcome.value,
                    input_count=1, output_count=0,
                )
                return QueryResult(
                    trace_id=ctx.trace_id, query=query, broker_result=broker_result,
                    total_latency_ms=_elapsed_ms(start), outcome=broker_result.outcome.value,
                )
            candidates = fused_list_to_retrieval_candidates(broker_result.candidates)
            reranker_result = self._run_reranking(ctx, candidates, rq)
            record_stage_result(
                root, outcome="success",
                input_count=1, output_count=reranker_result.candidate_count,
            )
            return QueryResult(
                trace_id=ctx.trace_id, query=query, broker_result=broker_result,
                reranker_result=reranker_result, total_latency_ms=_elapsed_ms(start),
                outcome="success",
            )

    def run_through_context(
        self,
        query: str,
        trace_id: str | None = None,
        top_k: int = 50,
        token_budget: int | None = None,
    ) -> QueryResult:
        """Run retrieval + reranking + context building."""
        start = time.monotonic()
        ctx = self._tracer.create_context(trace_id=trace_id)
        budget = token_budget or self._token_budget
        with pipeline_span(self._tracer, ctx, "rag-pipeline") as root:
            root.set_attribute("query", query)
            rq = RetrievalQuery(
                raw_query=query, normalized_query=query,
                trace_id=ctx.trace_id, top_k=top_k,
            )
            broker_result = self._run_retrieval(ctx, rq)
            if broker_result.outcome in (BrokerOutcome.FAILED, BrokerOutcome.NO_RESULTS):
                record_stage_result(
                    root, outcome=broker_result.outcome.value,
                    input_count=1, output_count=0,
                )
                return QueryResult(
                    trace_id=ctx.trace_id, query=query, broker_result=broker_result,
                    total_latency_ms=_elapsed_ms(start), outcome=broker_result.outcome.value,
                )
            candidates = fused_list_to_retrieval_candidates(broker_result.candidates)
            reranker_result = self._run_reranking(ctx, candidates, rq)
            builder_result = self._run_context_build(
                ctx, reranker_result.ranked_candidates, query, budget,
            )
            record_stage_result(
                root, outcome="success", input_count=1,
                output_count=len(builder_result.context_pack.evidence),
            )
            return QueryResult(
                trace_id=ctx.trace_id, query=query, broker_result=broker_result,
                reranker_result=reranker_result, builder_result=builder_result,
                total_latency_ms=_elapsed_ms(start), outcome="success",
            )

    # --- Internal stage runners ---

    def _run_retrieval(
        self, ctx: ObservabilityContext, query: RetrievalQuery,
    ) -> BrokerResult:
        with pipeline_span(self._tracer, ctx, "retrieval") as span:
            result = self._broker.run(query)
            record_stage_result(
                span, outcome=result.outcome.value,
                input_count=1, output_count=result.candidate_count,
            )
            return result

    def _run_reranking(
        self, ctx: ObservabilityContext,
        candidates: list[RetrievalCandidate], query: RetrievalQuery,
    ) -> RerankerResult:
        with pipeline_span(self._tracer, ctx, "reranking") as span:
            result = self._reranker.run(candidates, query)
            record_stage_result(
                span, outcome=result.outcome.value,
                input_count=len(candidates), output_count=result.candidate_count,
            )
            return result

    def _run_context_build(
        self, ctx: ObservabilityContext,
        ranked: list[RankedCandidate], query: str, token_budget: int,
    ) -> BuilderResult:
        with pipeline_span(self._tracer, ctx, "context_build") as span:
            result = self._context_builder.run(ranked, query, token_budget)
            record_stage_result(
                span, outcome=result.outcome.value,
                input_count=len(ranked),
                output_count=len(result.context_pack.evidence),
            )
            return result

    def _run_generation(
        self,
        ctx: ObservabilityContext,
        builder_result: BuilderResult,
        trace_id: str,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> GenerationResult:
        with pipeline_span(self._tracer, ctx, "generation") as span:
            result = self._generator.run(
                builder_result.context_pack, trace_id, on_token=on_token,
            )
            record_stage_result(
                span, outcome=result.outcome.value,
                input_count=len(builder_result.context_pack.evidence),
                output_count=1 if result.answer else 0,
            )
            # Enrich span with LLM metadata for observability exporters
            span.set_attribute("generator_id", result.generator_id)
            if result.answer is not None:
                span.set_attribute("model_id", result.answer.model_id)
                span.set_attribute("prompt_tokens", result.answer.token_usage.prompt_tokens)
                span.set_attribute("completion_tokens", result.answer.token_usage.completion_tokens)
            return result


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
