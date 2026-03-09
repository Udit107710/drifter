"""Metric abstractions compatible with OpenTelemetry metrics API."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class CounterMetric:
    """A monotonically increasing counter."""
    name: str
    description: str = ""
    _value: float = field(default=0.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def increment(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        """Reset for testing only."""
        with self._lock:
            self._value = 0.0


@dataclass
class HistogramMetric:
    """Records a distribution of values (e.g., latencies)."""
    name: str
    description: str = ""
    unit: str = "ms"
    _values: list[float] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values) if self._values else 0.0

    @property
    def values(self) -> list[float]:
        return list(self._values)

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100). Returns 0.0 if empty."""
        if not self._values:
            return 0.0
        sorted_vals = sorted(self._values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def reset(self) -> None:
        """Reset for testing only."""
        with self._lock:
            self._values.clear()


# Pre-defined pipeline metrics
pipeline_latency = HistogramMetric(
    name="drifter.pipeline.stage.latency",
    description="Latency per pipeline stage",
    unit="ms",
)

pipeline_errors = CounterMetric(
    name="drifter.pipeline.stage.errors",
    description="Error count per pipeline stage",
)

pipeline_throughput = CounterMetric(
    name="drifter.pipeline.throughput",
    description="Items processed per stage",
)
