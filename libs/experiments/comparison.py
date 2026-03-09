"""Experiment comparison: metric deltas and config diffs."""

from __future__ import annotations

from dataclasses import asdict

from libs.experiments.models import (
    ExperimentComparison,
    ExperimentRun,
    MetricDelta,
)


def compare_runs(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
) -> ExperimentComparison:
    """Compare two experiment runs, computing metric deltas and config diffs."""
    # Compute metric deltas
    baseline_metrics = baseline.report.aggregate_metrics
    candidate_metrics = candidate.report.aggregate_metrics
    all_keys = sorted(set(baseline_metrics) | set(candidate_metrics))

    deltas: list[MetricDelta] = []
    for key in all_keys:
        bv = baseline_metrics.get(key, 0.0)
        cv = candidate_metrics.get(key, 0.0)
        absolute = cv - bv
        relative = absolute / bv if bv != 0 else 0.0
        deltas.append(MetricDelta(
            metric_name=key,
            baseline_value=bv,
            candidate_value=cv,
            absolute_change=absolute,
            relative_change=relative,
            improved=cv > bv,
        ))

    # Compute config diffs
    baseline_cfg = asdict(baseline.config.eval_config)
    candidate_cfg = asdict(candidate.config.eval_config)
    config_diffs: dict[str, tuple[str, str]] = {}
    all_cfg_keys = sorted(set(baseline_cfg) | set(candidate_cfg))
    for key in all_cfg_keys:
        bv_str = str(baseline_cfg.get(key, ""))
        cv_str = str(candidate_cfg.get(key, ""))
        if bv_str != cv_str:
            config_diffs[key] = (bv_str, cv_str)

    return ExperimentComparison(
        baseline=baseline,
        candidate=candidate,
        deltas=deltas,
        config_diffs=config_diffs,
    )


def generate_comparison_markdown(comparison: ExperimentComparison) -> str:
    """Generate a human-readable markdown comparison report."""
    lines: list[str] = []
    lines.append("# Experiment Comparison")
    lines.append("")
    lines.append(
        f"- **Baseline**: {comparison.baseline.run_id} "
        f"({comparison.baseline.config.name})"
    )
    lines.append(
        f"- **Candidate**: {comparison.candidate.run_id} "
        f"({comparison.candidate.config.name})"
    )
    lines.append("")

    # Config diffs
    if comparison.config_diffs:
        lines.append("## Configuration Differences")
        lines.append("")
        lines.append("| Parameter | Baseline | Candidate |")
        lines.append("|-----------|----------|-----------|")
        for key, (bv, cv) in sorted(comparison.config_diffs.items()):
            lines.append(f"| {key} | {bv} | {cv} |")
        lines.append("")

    # Metric deltas
    lines.append("## Metric Deltas")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Change | Relative | Improved |")
    lines.append("|--------|----------|-----------|--------|----------|----------|")
    for d in comparison.deltas:
        sign = "+" if d.absolute_change >= 0 else ""
        improved = "yes" if d.improved else "no"
        lines.append(
            f"| {d.metric_name} "
            f"| {d.baseline_value:.4f} "
            f"| {d.candidate_value:.4f} "
            f"| {sign}{d.absolute_change:.4f} "
            f"| {d.relative_change:+.1%} "
            f"| {improved} |"
        )
    lines.append("")

    return "\n".join(lines)
