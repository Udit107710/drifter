"""drifter retrieve — run retrieval only."""

from __future__ import annotations

import argparse

from apps.cli.errors import EXIT_INPUT_ERROR, outcome_to_exit_code
from apps.cli.output import OutputRenderer
from libs.retrieval.broker.models import BrokerConfig, RetrievalMode
from orchestrators.bootstrap import ServiceRegistry
from orchestrators.query import QueryOrchestrator


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("retrieve", help="Run retrieval only")
    p.add_argument("query", help="Query text")
    p.add_argument("--top-k", type=int, default=10, help="Number of candidates")
    p.add_argument(
        "--mode",
        choices=["dense", "lexical", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    if not args.query.strip():
        renderer.render_error("Query must not be empty")
        return EXIT_INPUT_ERROR

    # Apply mode override to broker config
    mode = RetrievalMode(args.mode)
    registry.retrieval_broker._config = BrokerConfig(mode=mode)

    orchestrator = QueryOrchestrator(
        tracer=registry.tracer,
        retrieval_broker=registry.retrieval_broker,
        reranker_service=registry.reranker_service,
        context_builder_service=registry.context_builder_service,
        generation_service=registry.generation_service,
        token_budget=registry.token_budget,
    )

    result = orchestrator.run_retrieve_only(
        query=args.query,
        trace_id=getattr(args, "trace", None),
        top_k=args.top_k,
    )

    if result.broker_result:
        renderer.render_broker_result(result.broker_result, result.trace_id)

    return outcome_to_exit_code(result.outcome)
