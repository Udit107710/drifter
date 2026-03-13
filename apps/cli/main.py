"""CLI entry point for Drifter RAG system.

Usage: drifter <command> [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from apps.cli.commands import (
    ask,
    context,
    debug_query,
    evaluate,
    experiment,
    generate,
    index,
    ingest,
    rerank,
    retrieve,
)
from apps.cli.errors import EXIT_CONFIG_ERROR, EXIT_INPUT_ERROR
from apps.cli.output import OutputRenderer
from orchestrators.bootstrap import create_registry

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="drifter",
        description="Drifter RAG System CLI",
    )

    # Global flags
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--trace", default=None, help="Use specific trace ID")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config (repeatable)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file (default: ./.env)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Register all commands
    retrieve.register(subparsers)
    rerank.register(subparsers)
    context.register(subparsers)
    generate.register(subparsers)
    ask.register(subparsers)
    debug_query.register(subparsers)
    ingest.register(subparsers)
    index.register(subparsers)
    evaluate.register(subparsers)
    experiment.register(subparsers)

    return parser


def _parse_config_overrides(config_args: list[str]) -> dict[str, str]:
    """Parse --config key=value pairs into a dict."""
    overrides: dict[str, str] = {}
    for item in config_args:
        if "=" not in item:
            raise ValueError(f"Invalid config format: {item!r}. Use KEY=VALUE.")
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    # Pre-parse --env-file so we can load it before full arg parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    # Load .env (explicit path, or default from project root / CWD)
    _env_file = Path(pre_args.env_file) if pre_args.env_file else _PROJECT_ROOT / ".env"
    load_dotenv(_env_file)

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT_INPUT_ERROR

    renderer = OutputRenderer(
        json_mode=args.json,
        verbose=args.verbose,
    )

    # Parse config overrides
    try:
        overrides = _parse_config_overrides(args.config)
    except ValueError as exc:
        renderer.render_error(str(exc))
        return EXIT_CONFIG_ERROR

    # Create service registry
    try:
        registry = create_registry(
            overrides=overrides if overrides else None,
            config_path=args.config_file,
        )
    except ValueError as exc:
        renderer.render_error(str(exc))
        return EXIT_CONFIG_ERROR
    except Exception as exc:
        renderer.render_error(f"Failed to initialize services: {exc}")
        return EXIT_CONFIG_ERROR

    # Dispatch to command handler
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_INPUT_ERROR

    result: int = handler(args, registry, renderer)
    return result


if __name__ == "__main__":
    sys.exit(main())
