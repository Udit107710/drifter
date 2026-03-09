"""Span collectors: receive completed spans for export or analysis."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.observability.spans import Span


@runtime_checkable
class SpanCollector(Protocol):
    """Protocol for receiving completed spans."""

    def collect(self, span: Span) -> None: ...


class InMemoryCollector:
    """Collects spans in memory for testing and debugging."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def collect(self, span: Span) -> None:
        self._spans.append(span)

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def find_by_name(self, name: str) -> list[Span]:
        return [s for s in self._spans if s.name == name]

    def find_by_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def clear(self) -> None:
        self._spans.clear()

    @property
    def count(self) -> int:
        return len(self._spans)


class NoOpCollector:
    """Discards all spans. Used when observability is disabled."""

    def collect(self, span: Span) -> None:
        pass
