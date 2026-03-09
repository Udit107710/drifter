"""rag evaluate — run retrieval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from apps.cli.errors import EXIT_CONFIG_ERROR, EXIT_FAILED, EXIT_INPUT_ERROR, EXIT_SUCCESS
from apps.cli.output import OutputRenderer
from libs.evaluation.dataset import load_dataset
from libs.retrieval.broker.service import RetrievalBroker
from orchestrators.bootstrap import ServiceRegistry


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("evaluate", help="Run retrieval evaluation")
    p.add_argument("--dataset", required=True, help="Path to evaluation dataset JSON")
    p.add_argument(
        "--k",
        default="5,10,20",
        help="Comma-separated k values (default: 5,10,20)",
    )
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        renderer.render_error(f"Dataset file not found: {dataset_path}")
        return EXIT_INPUT_ERROR

    try:
        k_values = [int(k.strip()) for k in args.k.split(",")]
    except ValueError:
        renderer.render_error(f"Invalid k values: {args.k}")
        return EXIT_INPUT_ERROR

    try:
        cases = load_dataset(dataset_path)
    except Exception as exc:
        renderer.render_error(f"Failed to load dataset: {exc}")
        return EXIT_CONFIG_ERROR

    # Create a simple retriever that uses the broker
    class _BrokerRetriever:
        def __init__(self, broker: RetrievalBroker) -> None:
            self._broker = broker

        def retrieve(self, query: str, k: int) -> list[str]:
            from libs.contracts.retrieval import RetrievalQuery
            rq = RetrievalQuery(
                raw_query=query, normalized_query=query,
                trace_id="eval", top_k=k,
            )
            result = self._broker.run(rq)
            return [c.chunk.chunk_id for c in result.candidates[:k]]

    retriever = _BrokerRetriever(registry.retrieval_broker)

    try:
        report = registry.evaluator.evaluate(
            cases=cases,
            retriever=retriever,
            k_values=k_values,
        )
    except Exception as exc:
        renderer.render_error(f"Evaluation failed: {exc}")
        return EXIT_FAILED

    renderer.render_evaluation_report(report)
    return EXIT_SUCCESS
