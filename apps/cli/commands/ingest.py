"""rag ingest — run ingestion pipeline on local files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from libs.ingestion.models import SourceConfig, SourceType

from apps.cli.errors import EXIT_CONFIG_ERROR, EXIT_FAILED, EXIT_SUCCESS
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import ServiceRegistry


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ingest", help="Ingest documents from a local path")
    p.add_argument("--path", required=True, help="File or directory to ingest")
    p.add_argument("--run-id", default=None, help="Explicit run ID for replay detection")
    p.set_defaults(handler=handle)


def handle(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    orchestrator = registry.ingestion_orchestrator
    if orchestrator is None:
        renderer.render_error("Ingestion orchestrator not available.")
        return EXIT_CONFIG_ERROR

    source_repo = registry.source_repo
    target = Path(args.path).resolve()

    if not target.exists():
        renderer.render_error(f"Path does not exist: {target}")
        return EXIT_FAILED

    # Register sources for each file
    if target.is_file():
        files = [target]
    else:
        files = sorted(p for p in target.rglob("*") if p.is_file())

    if not files:
        renderer.render_error(f"No files found at: {target}")
        return EXIT_FAILED

    for f in files:
        source_id = f"file:{f.name}"
        config = SourceConfig(
            source_id=source_id,
            uri=str(f),
            source_type=SourceType.FILESYSTEM,
            enabled=True,
        )
        source_repo.add(config)

    # Run the pipeline
    trace_id = getattr(args, "trace", None)
    result = orchestrator.run(run_id=args.run_id, trace_id=trace_id)

    # Output
    if renderer._json_mode:
        renderer._emit_json({
            "trace_id": result.trace_id,
            "run_id": result.run_id,
            "documents_ingested": result.documents_ingested,
            "chunks_produced": result.chunks_produced,
            "chunks_indexed": result.chunks_indexed,
            "errors": result.errors,
            "total_latency_ms": round(result.total_latency_ms, 1),
        })
    else:
        print(f"trace: {result.trace_id}", file=sys.stderr)
        print(f"run_id: {result.run_id}", file=sys.stderr)
        print(f"latency: {result.total_latency_ms:.1f}ms", file=sys.stderr)
        print(file=sys.stderr)
        print(f"Documents ingested: {result.documents_ingested}")
        print(f"Chunks produced:    {result.chunks_produced}")
        print(f"Chunks indexed:     {result.chunks_indexed}")
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for e in result.errors:
                print(f"  - {e}")

    return EXIT_SUCCESS if not result.errors else EXIT_FAILED
