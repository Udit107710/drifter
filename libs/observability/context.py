"""Observability context for trace propagation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def generate_trace_id() -> str:
    """Generate a new trace ID (UUID4 hex, OTel-compatible length)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a new span ID (first 16 chars of UUID4 hex)."""
    return uuid.uuid4().hex[:16]


@dataclass
class ObservabilityContext:
    """Carries trace context through the pipeline.

    Each pipeline stage receives this context and can create child spans.
    The span_stack tracks nesting for automatic parent_span_id assignment.
    """
    trace_id: str = field(default_factory=generate_trace_id)
    span_stack: list[str] = field(default_factory=list)
    baggage: dict[str, str] = field(default_factory=dict)

    @property
    def current_span_id(self) -> str | None:
        """The ID of the innermost active span, or None."""
        return self.span_stack[-1] if self.span_stack else None

    def push_span(self, span_id: str) -> None:
        self.span_stack.append(span_id)

    def pop_span(self) -> str | None:
        return self.span_stack.pop() if self.span_stack else None
