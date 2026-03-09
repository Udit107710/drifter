"""OpenTelemetry-compatible span abstractions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SpanStatus(Enum):
    """Span completion status, matching OTel conventions."""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class SpanKind(Enum):
    """Span kind, matching OTel conventions."""
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class Span:
    """A single instrumentation span representing a unit of work.

    Mutable during its lifetime (attributes, events, status can be added).
    Becomes immutable once ended.
    """
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.monotonic)
    start_wall: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: float | None = None
    end_wall: datetime | None = None
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds. Returns 0.0 if not yet ended."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def is_ended(self) -> bool:
        return self.end_time is not None

    def set_attribute(self, key: str, value: Any) -> None:
        if self.is_ended:
            return
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        if self.is_ended:
            return
        self.events.append({
            "name": name,
            "timestamp": datetime.now(UTC).isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        if self.is_ended:
            return
        self.status = status
        if message:
            self.error_message = message

    def end(self) -> None:
        if self.is_ended:
            return
        self.end_time = time.monotonic()
        self.end_wall = datetime.now(UTC)
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """Export span as a dictionary (for JSON serialization / OTel export)."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "start_wall": self.start_wall.isoformat(),
            "end_wall": self.end_wall.isoformat() if self.end_wall else None,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }
