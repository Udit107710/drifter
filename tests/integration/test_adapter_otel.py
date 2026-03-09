"""Integration tests for OtelSpanExporter against a real Jaeger instance.

Run with: uv run pytest tests/integration/test_adapter_otel.py -v
Requires: docker compose up -d jaeger
"""

from __future__ import annotations

import json
import socket
import time
import uuid
from urllib.request import Request, urlopen

import pytest

from libs.adapters.config import OtelConfig
from libs.adapters.otel import OtelSpanExporter
from libs.observability.spans import Span, SpanKind, SpanStatus


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


OTEL_CONFIG = OtelConfig(
    endpoint="http://localhost:4318",
    protocol="http/protobuf",
    service_name="drifter-integration-test",
    export_interval_ms=500,
    insecure=True,
)


@pytest.fixture()
def otel_exporter():
    if not _port_open("localhost", 4318):
        pytest.skip("OTLP HTTP not available on localhost:4318")

    exporter = OtelSpanExporter(OTEL_CONFIG)
    exporter.connect()

    yield exporter

    exporter.close()


class TestOtelSpanExporter:
    def test_health_check(self, otel_exporter: OtelSpanExporter) -> None:
        assert otel_exporter.health_check() is True

    def test_collect_span(self, otel_exporter: OtelSpanExporter) -> None:
        """Export a span and verify it doesn't raise."""
        span = Span(
            name="test-span",
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            kind=SpanKind.INTERNAL,
            status=SpanStatus.OK,
            attributes={"query": "hello", "top_k": 10},
        )
        span.end()

        # Should not raise
        otel_exporter.collect(span)

    def test_collect_error_span(self, otel_exporter: OtelSpanExporter) -> None:
        """Export an error span with events."""
        span = Span(
            name="error-span",
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            kind=SpanKind.CLIENT,
            attributes={"component": "retrieval"},
        )
        span.add_event("retry", {"attempt": 1})
        span.set_status(SpanStatus.ERROR, "connection timeout")
        span.end()

        otel_exporter.collect(span)

    def test_spans_reach_jaeger(self, otel_exporter: OtelSpanExporter) -> None:
        """Export a span and verify it shows up in Jaeger's API."""
        if not _port_open("localhost", 16686):
            pytest.skip("Jaeger UI not available on localhost:16686")

        unique_name = f"jaeger-verify-{uuid.uuid4().hex[:8]}"
        span = Span(
            name=unique_name,
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            kind=SpanKind.INTERNAL,
            status=SpanStatus.OK,
            attributes={"test.marker": unique_name},
        )
        span.end()
        otel_exporter.collect(span)

        # Force flush by closing and reopening
        otel_exporter.close()
        time.sleep(2)

        # Check Jaeger for the service
        req = Request("http://localhost:16686/api/services")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            services = data.get("data", [])
            assert "drifter-integration-test" in services, (
                f"Service 'drifter-integration-test' not found in Jaeger. "
                f"Found: {services}"
            )

    def test_collect_with_parent_span(self, otel_exporter: OtelSpanExporter) -> None:
        """Export a span with a parent relationship."""
        trace_id = str(uuid.uuid4())
        parent = Span(
            name="parent-span",
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            kind=SpanKind.SERVER,
        )
        parent.end()
        otel_exporter.collect(parent)

        child = Span(
            name="child-span",
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id,
            kind=SpanKind.INTERNAL,
        )
        child.end()
        otel_exporter.collect(child)
