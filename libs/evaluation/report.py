"""Report generation: JSON and markdown evaluation reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from libs.evaluation.models import EvaluationReport


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Convert report to a JSON-serializable dictionary."""
    data = asdict(report)
    # Convert datetime to ISO string
    data["evaluated_at"] = report.evaluated_at.isoformat()
    return data


def save_json_report(report: EvaluationReport, path: Path) -> None:
    """Save report as machine-readable JSON."""
    data = report_to_dict(report)
    path.write_text(json.dumps(data, indent=2, default=str))


def generate_markdown_summary(report: EvaluationReport) -> str:
    """Generate a human-readable markdown summary of evaluation results."""
    lines: list[str] = []
    lines.append(f"# Evaluation Report: {report.run_id}")
    lines.append("")
    lines.append(f"- **Date**: {report.evaluated_at.isoformat()}")
    lines.append(f"- **Dataset size**: {report.dataset_size} queries")
    if report.dataset_name:
        lines.append(f"- **Dataset**: {report.dataset_name}")
    lines.append("")

    # Configuration
    lines.append("## Configuration")
    lines.append("")
    cfg = report.config
    if cfg.retrieval_mode:
        lines.append(f"- Retrieval mode: {cfg.retrieval_mode}")
    if cfg.embedding_model:
        lines.append(f"- Embedding model: {cfg.embedding_model}")
    if cfg.reranker_id:
        lines.append(f"- Reranker: {cfg.reranker_id}")
    if cfg.chunking_strategy:
        lines.append(f"- Chunking: {cfg.chunking_strategy}")
    lines.append(f"- k values: {cfg.k_values}")
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate Metrics")
    lines.append("")
    for sm in report.stage_metrics:
        lines.append(f"### {sm.stage.title()} (n={sm.query_count})")
        lines.append("")
        lines.append("| Metric | Mean | Median |")
        lines.append("|--------|------|--------|")
        for key in sorted(sm.metric_means.keys()):
            mean = sm.metric_means[key]
            median = sm.metric_medians.get(key, 0.0)
            lines.append(f"| {key} | {mean:.4f} | {median:.4f} |")
        lines.append("")

    # Per-query breakdown
    lines.append("## Per-Query Results")
    lines.append("")
    for qr in report.query_results:
        lines.append(f"### {qr.case_id}: {qr.query}")
        lines.append("")
        hits = set(qr.retrieved_ids) & set(qr.relevant_ids)
        lines.append(f"- Retrieved: {len(qr.retrieved_ids)} | "
                     f"Relevant: {len(qr.relevant_ids)} | "
                     f"Hits: {len(hits)}")
        for key in sorted(qr.metrics.keys()):
            lines.append(f"- {key}: {qr.metrics[key]:.4f}")
        lines.append("")

    return "\n".join(lines)


def save_markdown_report(report: EvaluationReport, path: Path) -> None:
    """Save report as human-readable markdown."""
    md = generate_markdown_summary(report)
    path.write_text(md)
