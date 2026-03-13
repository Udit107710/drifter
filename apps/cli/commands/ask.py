"""drifter ask — full end-to-end pipeline, display answer."""

from __future__ import annotations

import argparse
import sys

from apps.cli.errors import EXIT_INPUT_ERROR, outcome_to_exit_code
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import ServiceRegistry
from orchestrators.query import QueryOrchestrator


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ask", help="Ask a question (full pipeline)")
    p.add_argument("query", help="Query text")
    p.add_argument("--top-k", type=int, default=10, help="Retrieval candidates")
    p.add_argument("--token-budget", type=int, default=0, help="Token budget override")
    p.add_argument(
        "--mode",
        choices=["dense", "lexical", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Stream thinking/answer tokens in real time",
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

    budget = args.token_budget if args.token_budget > 0 else None

    if hasattr(args, "mode") and args.mode:
        from libs.retrieval.broker.models import BrokerConfig, RetrievalMode
        registry.retrieval_broker._config = BrokerConfig(mode=RetrievalMode(args.mode))

    orchestrator = QueryOrchestrator(
        tracer=registry.tracer,
        retrieval_broker=registry.retrieval_broker,
        reranker_service=registry.reranker_service,
        context_builder_service=registry.context_builder_service,
        generation_service=registry.generation_service,
        token_budget=registry.token_budget,
    )

    # Set up streaming callback if requested
    on_token = None
    if getattr(args, "stream", False) and not getattr(args, "json", False):
        on_token = _make_stream_callback()

    result = orchestrator.run(
        query=args.query,
        trace_id=getattr(args, "trace", None),
        top_k=args.top_k,
        token_budget=budget,
        on_token=on_token,
    )

    # If we were streaming, add a newline after the streamed output
    if on_token is not None:
        print(file=sys.stderr)  # newline after thinking
        # Don't re-print the answer since it was already streamed
        if result.outcome in ("success", "partial"):
            renderer._meta(f"trace: {result.trace_id}")
            renderer._meta(f"outcome: {result.outcome}")
            renderer._meta(f"latency: {result.total_latency_ms:.1f}ms")
            if result.errors:
                for e in result.errors:
                    renderer._meta(f"warning: {e}")
            return outcome_to_exit_code(result.outcome)

    renderer.render_query_result(result)
    return outcome_to_exit_code(result.outcome)


def _make_stream_callback():
    """Create a streaming callback that prints tokens to stderr/stdout."""
    in_thinking = [False]

    def on_token(text: str, is_thinking: bool) -> None:
        if is_thinking:
            if not in_thinking[0]:
                print("\n[thinking] ", end="", file=sys.stderr, flush=True)
                in_thinking[0] = True
            print(text, end="", file=sys.stderr, flush=True)
        else:
            if in_thinking[0]:
                print("\n\n", end="", file=sys.stderr, flush=True)
                in_thinking[0] = False
            print(text, end="", flush=True)

    return on_token
