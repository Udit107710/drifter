"""Async query orchestrator: async retrieval, sync reranking/context/generation.

Reuses the shared stage runners (reranking, context building, generation)
from ``QueryOrchestrator`` — only the retrieval stage differs (async fanout
via ``AsyncRetrievalBroker``).
"""

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
from libs.retrieval.broker.models import BrokerOutcome, BrokerResult
from libs.retrieval.broker.service import RetrievalBroker
from orchestrators.query import QueryOrchestrator, QueryResult

logger = logging.getLogger(__name__)


class AsyncQueryOrchestrator(QueryOrchestrator):
    """Async pipeline: async retrieval, then delegates stages 2-4 to the
    sync runners inherited from ``QueryOrchestrator``."""

    def __init__(
        self,
        tracer: Tracer,
        async_retrieval_broker: AsyncRetrievalBroker,
        reranker_service: RerankerService,
        context_builder_service: ContextBuilderService,
        generation_service: GenerationService,
        token_budget: int = 3000,
        retrieval_broker: RetrievalBroker | None = None,
    ) -> None:
        # Supply a no-op sync broker when none given — async path never uses it.
        if retrieval_broker is None:
            retrieval_broker = _NoOpBroker()  # type: ignore[assignment]
        super().__init__(
            tracer=tracer,
            retrieval_broker=retrieval_broker,
            reranker_service=reranker_service,
            context_builder_service=context_builder_service,
            generation_service=generation_service,
            token_budget=token_budget,
        )
        self._async_broker = async_retrieval_broker

    async def async_run(
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
            broker_result = await self._async_broker.run(retrieval_query)

            if broker_result.outcome == BrokerOutcome.FAILED:
                record_stage_result(
                    root, outcome="failed",
                    input_count=1, output_count=0,
                )
                return QueryResult(
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

            if broker_result.outcome == BrokerOutcome.NO_RESULTS:
                record_stage_result(
                    root, outcome="no_results",
                    input_count=1, output_count=0,
                )
                return QueryResult(
                    trace_id=ctx.trace_id,
                    query=query,
                    broker_result=broker_result,
                    total_latency_ms=_elapsed_ms(start),
                    outcome="no_results",
                )

            # Stages 2-4: Reuse sync runners from QueryOrchestrator
            errors: list[str] = []
            candidates = fused_list_to_retrieval_candidates(
                broker_result.candidates,
            )

            reranker_result = None
            try:
                reranker_result = self._run_reranking(
                    ctx, candidates, retrieval_query,
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

            builder_result = self._run_context_build(
                ctx, ranked, query, budget,
            )
            generation_result = self._run_generation(
                ctx, builder_result, ctx.trace_id,
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


class _NoOpBroker:
    """Placeholder broker that is never called — satisfies the parent __init__."""

    def run(self, query: RetrievalQuery) -> BrokerResult:
        raise RuntimeError("_NoOpBroker.run() should never be called")


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
