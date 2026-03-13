"""drifter experiment — run experiments and compare runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.cli.errors import EXIT_CONFIG_ERROR, EXIT_FAILED, EXIT_INPUT_ERROR, EXIT_SUCCESS
from apps.cli.output import OutputRenderer
from libs.evaluation.models import EvaluationConfig
from libs.experiments.models import ExperimentConfig
from libs.retrieval.broker.service import RetrievalBroker
from orchestrators.bootstrap import ServiceRegistry


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("experiment", help="Run experiments")
    sub = p.add_subparsers(dest="experiment_command")

    run_p = sub.add_parser("run", help="Run an experiment")
    run_p.add_argument("--config", required=True, dest="exp_config", help="Experiment config JSON")
    run_p.add_argument("--hypothesis", default="", help="Experiment hypothesis")
    run_p.set_defaults(handler=handle_run)

    compare_p = sub.add_parser("compare", help="Compare two runs")
    compare_p.add_argument("--baseline", required=True, help="Baseline run ID")
    compare_p.add_argument("--candidate", required=True, help="Candidate run ID")
    compare_p.set_defaults(handler=handle_compare)


def handle_run(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    config_path = Path(args.exp_config)
    if not config_path.exists():
        renderer.render_error(f"Config file not found: {config_path}")
        return EXIT_INPUT_ERROR

    try:
        raw = json.loads(config_path.read_text())
    except Exception as exc:
        renderer.render_error(f"Failed to read config: {exc}")
        return EXIT_CONFIG_ERROR

    try:
        eval_config = EvaluationConfig(**raw.get("eval_config", {}))
        exp_config = ExperimentConfig(
            name=raw.get("name", config_path.stem),
            hypothesis=args.hypothesis or raw.get("hypothesis", ""),
            eval_config=eval_config,
            dataset_path=raw.get("dataset_path", ""),
            artifact_dir=raw.get("artifact_dir", "artifacts"),
            k_values=raw.get("k_values", [5, 10, 20]),
            tags=raw.get("tags", []),
            notes=raw.get("notes", ""),
        )
    except Exception as exc:
        renderer.render_error(f"Invalid config: {exc}")
        return EXIT_CONFIG_ERROR

    # Create retriever from broker
    class _BrokerRetriever:
        def __init__(self, broker: RetrievalBroker) -> None:
            self._broker = broker

        def retrieve(self, query: str, k: int) -> list[str]:
            from libs.contracts.retrieval import RetrievalQuery
            rq = RetrievalQuery(
                raw_query=query, normalized_query=query,
                trace_id="experiment", top_k=k,
            )
            result = self._broker.run(rq)
            return [c.chunk.chunk_id for c in result.candidates[:k]]

    retriever = _BrokerRetriever(registry.retrieval_broker)

    try:
        run = registry.experiment_runner.run(
            config=exp_config,
            retriever=retriever,
        )
    except Exception as exc:
        renderer.render_error(f"Experiment failed: {exc}")
        return EXIT_FAILED

    renderer.render_evaluation_report(run.report)
    return EXIT_SUCCESS


def handle_compare(
    args: argparse.Namespace,
    registry: ServiceRegistry,
    renderer: OutputRenderer,
) -> int:
    renderer.render_error(
        "Experiment comparison requires a persistent experiment store. "
        "This feature will be available once database storage is configured."
    )
    return EXIT_FAILED
