"""OpenTelemetry span exporter adapter.

Converts Drifter spans to OTel spans and exports them via OTLP HTTP
to a collector (e.g. Jaeger, Grafana Tempo).
"""

from __future__ import annotations

import logging
from typing import Any

from libs.adapters.config import OtelConfig
from libs.observability.spans import (
    Span,
)
from libs.observability.spans import (
    SpanKind as DrifterSpanKind,
)
from libs.observability.spans import (
    SpanStatus as DrifterSpanStatus,
)

logger = logging.getLogger(__name__)

# Allowed attribute value types for OTel spans.
_OTEL_ATTR_TYPES = (str, int, float, bool)


class OtelSpanExporter:
    """Span exporter that forwards Drifter spans to an OpenTelemetry collector.

    Satisfies the :class:`~libs.observability.collector.SpanCollector`
    protocol (single method: ``collect(span) -> None``).

    OTel SDK imports are deferred to :meth:`connect` so the adapter can
    be instantiated even when ``opentelemetry-sdk`` is not installed.
    """

    def __init__(self, config: OtelConfig) -> None:
        self._config = config
        self._provider: Any = None
        self._tracer: Any = None

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> None:
        """Set up the OTel TracerProvider, BatchSpanProcessor, and OTLP exporter."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise RuntimeError(
                "opentelemetry-sdk and opentelemetry-exporter-otlp-proto-http "
                "must be installed to use OtelSpanExporter. "
                "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
            ) from exc

        resource = Resource.create({"service.name": self._config.service_name})

        # OTLP HTTP exporter pointed at the collector endpoint.
        endpoint = self._config.endpoint.rstrip("/") + "/v1/traces"
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)

        processor = BatchSpanProcessor(
            otlp_exporter,
            schedule_delay_millis=self._config.export_interval_ms,
        )

        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(processor)
        self._tracer = self._provider.get_tracer(
            "drifter.otel.exporter",
            schema_url="https://opentelemetry.io/schemas/1.21.0",
        )

        logger.info(
            "OtelSpanExporter connected — endpoint=%s service=%s",
            endpoint,
            self._config.service_name,
        )

    def close(self) -> None:
        """Flush pending spans and shut down the tracer provider."""
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception:
                logger.exception("Error shutting down OTel TracerProvider")
            finally:
                self._provider = None
                self._tracer = None

    def health_check(self) -> bool:
        """Return ``True`` if the tracer provider is initialised."""
        return self._provider is not None and self._tracer is not None

    # -- SpanCollector protocol ------------------------------------------------

    def collect(self, span: Span) -> None:
        """Convert a Drifter :class:`Span` to an OTel span and export it."""
        if self._tracer is None:
            raise RuntimeError(
                "OtelSpanExporter is not connected. Call connect() first."
            )

        from opentelemetry.trace import StatusCode

        otel_kind = _map_span_kind(span.kind)
        otel_span = self._tracer.start_span(name=span.name, kind=otel_kind)

        # --- Drifter identity attributes (OTel generates its own IDs) ---
        otel_span.set_attribute("drifter.trace_id", span.trace_id)
        otel_span.set_attribute("drifter.span_id", span.span_id)
        if span.parent_span_id is not None:
            otel_span.set_attribute("drifter.parent_span_id", span.parent_span_id)

        # --- Duration ---
        otel_span.set_attribute("drifter.duration_ms", span.duration_ms)

        # --- Wall-clock timestamps as ISO strings ---
        otel_span.set_attribute("drifter.start_wall", span.start_wall.isoformat())
        if span.end_wall is not None:
            otel_span.set_attribute("drifter.end_wall", span.end_wall.isoformat())

        # --- User-defined attributes (filter to OTel-safe types) ---
        for key, value in span.attributes.items():
            if isinstance(value, _OTEL_ATTR_TYPES):
                otel_span.set_attribute(key, value)
            else:
                # Coerce unsupported types to string so nothing is silently lost.
                otel_span.set_attribute(key, str(value))

        # --- Events ---
        for event in span.events:
            event_name = event.get("name", "event")
            event_attrs = event.get("attributes", {})
            # Filter event attributes to OTel-safe types as well.
            safe_attrs: dict[str, str | int | float | bool] = {}
            for k, v in event_attrs.items():
                safe_attrs[k] = v if isinstance(v, _OTEL_ATTR_TYPES) else str(v)
            # Include original timestamp as an attribute if present.
            if "timestamp" in event:
                safe_attrs["drifter.event.timestamp"] = str(event["timestamp"])
            otel_span.add_event(event_name, attributes=safe_attrs)

        # --- Error message ---
        if span.error_message is not None:
            otel_span.set_attribute("error.message", span.error_message)

        # --- Status ---
        status_code = _map_status(span.status)
        if status_code == StatusCode.ERROR:
            otel_span.set_status(status_code, span.error_message or "")
        else:
            otel_span.set_status(status_code)

        # --- End the span so the processor picks it up ---
        otel_span.end()


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _map_span_kind(kind: DrifterSpanKind) -> Any:
    """Map a Drifter SpanKind to an OTel SpanKind."""
    from opentelemetry.trace import SpanKind as OtelSpanKind

    _KIND_MAP = {
        DrifterSpanKind.INTERNAL: OtelSpanKind.INTERNAL,
        DrifterSpanKind.CLIENT: OtelSpanKind.CLIENT,
        DrifterSpanKind.SERVER: OtelSpanKind.SERVER,
        DrifterSpanKind.PRODUCER: OtelSpanKind.PRODUCER,
        DrifterSpanKind.CONSUMER: OtelSpanKind.CONSUMER,
    }
    return _KIND_MAP.get(kind, OtelSpanKind.INTERNAL)


def _map_status(status: DrifterSpanStatus) -> Any:
    """Map a Drifter SpanStatus to an OTel StatusCode."""
    from opentelemetry.trace import StatusCode

    _STATUS_MAP = {
        DrifterSpanStatus.OK: StatusCode.OK,
        DrifterSpanStatus.ERROR: StatusCode.ERROR,
        DrifterSpanStatus.UNSET: StatusCode.UNSET,
    }
    return _STATUS_MAP.get(status, StatusCode.UNSET)
