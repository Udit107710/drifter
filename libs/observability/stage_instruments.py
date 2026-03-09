"""High-level instrumentation helpers for pipeline stages."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from libs.observability.context import ObservabilityContext
from libs.observability.metrics import pipeline_errors, pipeline_latency, pipeline_throughput
from libs.observability.spans import Span, SpanKind
from libs.observability.tracer import Tracer


@contextmanager
def pipeline_span(
    tracer: Tracer,
    ctx: ObservabilityContext,
    stage: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Instrument a pipeline stage with span, metrics, and error tracking.

    Usage:
        with pipeline_span(tracer, ctx, "retrieval", attributes={"mode": "hybrid"}) as span:
            span.set_attribute("candidate_count", 42)
            result = do_retrieval(...)
        # Automatically: span ended, latency recorded, throughput incremented

    On error: increments error counter, sets span status to ERROR, re-raises.
    """
    with tracer.start_span(stage, ctx, kind=kind, attributes=attributes) as span:
        span.set_attribute("pipeline.stage", stage)
        try:
            yield span
            pipeline_throughput.increment()
        except Exception:
            pipeline_errors.increment()
            raise
        finally:
            # Record latency once span is ended (happens in tracer's finally)
            # We access duration_ms after end() is called by the tracer
            pass
    # Span is now ended by the tracer — record its latency
    pipeline_latency.record(span.duration_ms)


def record_stage_result(
    span: Span,
    *,
    outcome: str,
    input_count: int = 0,
    output_count: int = 0,
    error_count: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    """Record standard result attributes on a span.

    Call this before the pipeline_span context manager exits.
    """
    span.set_attribute("outcome", outcome)
    span.set_attribute("input_count", input_count)
    span.set_attribute("output_count", output_count)
    if error_count:
        span.set_attribute("error_count", error_count)
    if extra:
        for key, value in extra.items():
            span.set_attribute(key, value)
