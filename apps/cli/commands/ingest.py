"""rag ingest — run ingestion pipeline."""

from __future__ import annotations

import argparse

from apps.cli.errors import EXIT_FAILED
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import ServiceRegistry


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ingest", help="Run ingestion pipeline")
    p.add_argument("--run-id", default=None, help="Explicit run ID for replay detection")
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    renderer.render_error(
        "Ingestion requires a configured source connector. "
        "Set DRIFTER_* environment variables for your source."
    )
    return EXIT_FAILED
