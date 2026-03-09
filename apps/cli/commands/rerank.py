"""rag rerank — run retrieval + reranking."""

from __future__ import annotations

import argparse

from apps.cli.errors import EXIT_INPUT_ERROR, outcome_to_exit_code
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import ServiceRegistry
from orchestrators.query import QueryOrchestrator


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("rerank", help="Retrieve + rerank")
    p.add_argument("query", help="Query text")
    p.add_argument("--top-k", type=int, default=50, help="Retrieval candidates")
    p.add_argument("--top-n", type=int, default=0, help="Reranker top-n cutoff")
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    if not args.query.strip():
        renderer.render_error("Query must not be empty")
        return EXIT_INPUT_ERROR

    if args.top_n:
        registry.reranker_service._top_n = args.top_n

    orchestrator = QueryOrchestrator(
        tracer=registry.tracer,
        retrieval_broker=registry.retrieval_broker,
        reranker_service=registry.reranker_service,
        context_builder_service=registry.context_builder_service,
        generation_service=registry.generation_service,
        token_budget=registry.token_budget,
    )

    result = orchestrator.run_through_rerank(
        query=args.query,
        trace_id=getattr(args, "trace", None),
        top_k=args.top_k,
    )

    if result.reranker_result:
        renderer.render_reranker_result(result.reranker_result, result.trace_id)
    elif result.broker_result:
        renderer.render_broker_result(result.broker_result, result.trace_id)

    return outcome_to_exit_code(result.outcome)
