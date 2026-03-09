"""Tracer: creates and manages spans with context propagation."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from libs.observability.collector import NoOpCollector, SpanCollector
from libs.observability.context import ObservabilityContext, generate_span_id
from libs.observability.spans import Span, SpanKind, SpanStatus


class Tracer:
    """Creates spans and submits them to a collector on completion.

    Usage:
        tracer = Tracer(collector=InMemoryCollector())
        ctx = ObservabilityContext(trace_id="abc123")

        with tracer.start_span("retrieval", ctx) as span:
            span.set_attribute("candidate_count", 42)
            # ... do work ...
        # span is auto-ended and collected
    """

    def __init__(
        self,
        collector: SpanCollector | None = None,
        service_name: str = "drifter",
    ) -> None:
        self._collector = collector or NoOpCollector()
        self._service_name = service_name

    @property
    def service_name(self) -> str:
        return self._service_name

    @contextmanager
    def start_span(
        self,
        name: str,
        ctx: ObservabilityContext,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Create a span as a context manager.

        Automatically:
        - Sets parent_span_id from context stack
        - Pushes/pops span on context stack
        - Ends span and submits to collector
        - Sets ERROR status if exception occurs
        """
        span_id = generate_span_id()
        span = Span(
            name=name,
            trace_id=ctx.trace_id,
            span_id=span_id,
            parent_span_id=ctx.current_span_id,
            kind=kind,
            attributes=attributes or {},
        )
        span.set_attribute("service.name", self._service_name)

        ctx.push_span(span_id)
        try:
            yield span
        except Exception as exc:
            span.set_status(SpanStatus.ERROR, str(exc))
            span.add_event("exception", {
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
            })
            raise
        finally:
            ctx.pop_span()
            span.end()
            self._collector.collect(span)

    def create_context(self, trace_id: str | None = None) -> ObservabilityContext:
        """Create a new observability context, optionally with a given trace_id."""
        if trace_id:
            return ObservabilityContext(trace_id=trace_id)
        return ObservabilityContext()
