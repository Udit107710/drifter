"""drifter generate — full pipeline, show generation result."""

from __future__ import annotations

import argparse

from apps.cli.errors import EXIT_INPUT_ERROR, outcome_to_exit_code
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import ServiceRegistry
from orchestrators.query import QueryOrchestrator


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("generate", help="Full pipeline, show generation details")
    p.add_argument("query", help="Query text")
    p.add_argument("--top-k", type=int, default=10, help="Retrieval candidates")
    p.add_argument("--token-budget", type=int, default=0, help="Token budget override")
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    if not args.query.strip():
        renderer.render_error("Query must not be empty")
        return EXIT_INPUT_ERROR

    budget = args.token_budget if args.token_budget > 0 else None

    orchestrator = QueryOrchestrator(
        tracer=registry.tracer,
        retrieval_broker=registry.retrieval_broker,
        reranker_service=registry.reranker_service,
        context_builder_service=registry.context_builder_service,
        generation_service=registry.generation_service,
        token_budget=registry.token_budget,
    )

    result = orchestrator.run(
        query=args.query,
        trace_id=getattr(args, "trace", None),
        top_k=args.top_k,
        token_budget=budget,
    )

    if result.generation_result:
        renderer.render_generation_result(result.generation_result, result.trace_id)

    return outcome_to_exit_code(result.outcome)
