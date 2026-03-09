"""Async query orchestrator: async retrieval, sync reranking/context/generation."""

from __future__ import annotations

import logging
import time

from libs.context_builder.service import ContextBuilderService
from libs.contracts.retrieval import RankedCandidate, RetrievalQuery
from libs.generation.service import GenerationService
from libs.observability.stage_instruments import pipeline_span, record_stage_result
from libs.observability.tracer import Tracer
from libs.reranking.converters import fused_list_to_retrieval_candidates
from libs.reranking.service import RerankerService
from libs.retrieval.broker.async_service import AsyncRetrievalBroker
from libs.retrieval.broker.models import BrokerOutcome
from orchestrators.query import QueryResult

logger = logging.getLogger(__name__)


class AsyncQueryOrchestrator:
    """Async pipeline: async retrieval, sync reranking/context/generation."""

    def __init__(
        self,
        tracer: Tracer,
        async_retrieval_broker: AsyncRetrievalBroker,
        reranker_service: RerankerService,
        context_builder_service: ContextBuilderService,
        generation_service: GenerationService,
        token_budget: int = 3000,
    ) -> None:
        self._tracer = tracer
        self._broker = async_retrieval_broker
        self._reranker = reranker_service
        self._context_builder = context_builder_service
        self._generator = generation_service
        self._token_budget = token_budget

    async def run(
        self,
        query: str,
        trace_id: str | None = None,
        top_k: int = 50,
        token_budget: int | None = None,
    ) -> QueryResult:
        """Execute the full async query pipeline."""
        start = time.monotonic()
        ctx = self._tracer.create_context(trace_id=trace_id)
        budget = token_budget or self._token_budget

        logger.info(
            "async_pipeline: entry trace_id=%s query=%r top_k=%d",
            ctx.trace_id, query[:100], top_k,
        )

        with pipeline_span(self._tracer, ctx, "rag-pipeline") as root:
            root.set_attribute("query", query)
            root.set_attribute("async", True)

            retrieval_query = RetrievalQuery(
                raw_query=query,
                normalized_query=query,
                trace_id=ctx.trace_id,
                top_k=top_k,
            )

            # Stage 1: Async retrieval
            broker_result = await self._broker.run(retrieval_query)

            if broker_result.outcome == BrokerOutcome.FAILED:
                result = QueryResult(
                    trace_id=ctx.trace_id,
                    query=query,
                    broker_result=broker_result,
                    total_latency_ms=_elapsed_ms(start),
                    outcome="failed",
                    errors=[
                        "Retrieval failed: "
                        + "; ".join(broker_result.errors)
                    ],
                )
                record_stage_result(
                    root, outcome="failed",
                    input_count=1, output_count=0,
                )
                return result

            if broker_result.outcome == BrokerOutcome.NO_RESULTS:
                result = QueryResult(
                    trace_id=ctx.trace_id,
                    query=query,
                    broker_result=broker_result,
                    total_latency_ms=_elapsed_ms(start),
                    outcome="no_results",
                )
                record_stage_result(
                    root, outcome="no_results",
                    input_count=1, output_count=0,
                )
                return result

            # Stage 2: Sync reranking
            errors: list[str] = []
            candidates = fused_list_to_retrieval_candidates(
                broker_result.candidates,
            )
            reranker_result = None
            try:
                reranker_result = self._reranker.run(
                    candidates, retrieval_query,
                )
                ranked = reranker_result.ranked_candidates
            except Exception as exc:
                logger.warning(
                    "async_pipeline: reranking failed: %s", exc,
                )
                errors.append(
                    f"Reranking failed, using retrieval order: {exc}",
                )
                ranked = [
                    RankedCandidate(
                        candidate=c,
                        rank=i + 1,
                        rerank_score=c.score,
                        reranker_id="fallback",
                    )
                    for i, c in enumerate(candidates)
                ]

            # Stage 3: Sync context building
            builder_result = self._context_builder.run(
                ranked, query, budget,
            )

            # Stage 4: Sync generation
            generation_result = self._generator.run(
                builder_result.context_pack, ctx.trace_id,
            )

            outcome = "success" if not errors else "partial"
            record_stage_result(
                root, outcome=outcome,
                input_count=1,
                output_count=1 if outcome == "success" else 0,
            )

            result = QueryResult(
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

            logger.info(
                "async_pipeline: exit trace_id=%s outcome=%s "
                "latency=%.1fms",
                ctx.trace_id, result.outcome, result.total_latency_ms,
            )
            return result


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
