"""rag index — index previously ingested chunks."""

from __future__ import annotations

import argparse

from apps.cli.errors import EXIT_FAILED
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import ServiceRegistry


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("index", help="Index previously ingested chunks")
    p.add_argument("--run-id", default=None, help="Run ID to index chunks for")
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    if registry.indexing_service is None:
        renderer.render_error(
            "Indexing service not available. "
            "Configure embedding provider and storage backends."
        )
        return EXIT_FAILED

    renderer.render_error("Indexing requires previously ingested chunks. Use 'rag ingest' first.")
    return EXIT_FAILED
