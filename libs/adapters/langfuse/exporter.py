"""Langfuse span exporter adapter.

Converts Drifter spans into Langfuse traces, spans, and generations.
The ``generation`` pipeline stage is exported as a Langfuse generation
observation with model and token usage metadata; all other stages are
exported as plain Langfuse spans.

Implements the ``SpanCollector`` protocol.

Compatible with Langfuse SDK v3 (OTel-based API).

Architecture:
    Drifter's Tracer exports spans in end-order: child spans first, root span
    last.  The root ``rag-pipeline`` span (parent_span_id=None) signals that a
    complete trace is ready.  The exporter buffers child spans and flushes them
    all at once when the root arrives — root first — so Langfuse always names
    the trace correctly.

    Buffering can use either an in-memory dict (default) or Redis.  Redis
    buffering is recommended for production: it survives process restarts and
    works across multiple workers.  Set ``redis_url`` in :class:`LangfuseConfig`
    (via ``DRIFTER_LANGFUSE_REDIS_URL``) to enable it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from datetime import UTC
from typing import Any, Protocol

from libs.adapters.config import LangfuseConfig
from libs.observability.spans import Span, SpanKind, SpanStatus

logger = logging.getLogger(__name__)

# Pipeline stages that represent an LLM call.
_GENERATION_STAGES = frozenset({"generation"})

# Redis key prefix for buffered spans.
_REDIS_KEY_PREFIX = "drifter:langfuse:pending:"


# ---------------------------------------------------------------------------
# Span buffer protocol + implementations
# ---------------------------------------------------------------------------


class SpanBuffer(Protocol):
    """Abstract buffer for holding child spans until the root arrives."""

    def push(self, trace_id: str, span: Span) -> None: ...

    def pop_all(self, trace_id: str) -> list[Span]: ...

    def drain(self) -> list[Span]: ...

    def close(self) -> None: ...


class InMemorySpanBuffer:
    """Thread-safe in-memory buffer. Simple, zero-dependency default."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, list[Span]] = {}

    def push(self, trace_id: str, span: Span) -> None:
        with self._lock:
            self._pending.setdefault(trace_id, []).append(span)

    def pop_all(self, trace_id: str) -> list[Span]:
        with self._lock:
            return self._pending.pop(trace_id, [])

    def drain(self) -> list[Span]:
        with self._lock:
            all_spans: list[Span] = []
            for spans in self._pending.values():
                all_spans.extend(spans)
            self._pending.clear()
            return all_spans

    def close(self) -> None:
        pass


class RedisSpanBuffer:
    """Redis-backed buffer.  Survives restarts, works across workers.

    Each trace's child spans are stored as a Redis list at key
    ``drifter:langfuse:pending:<trace_id>``.  Keys auto-expire after
    ``ttl_s`` seconds to prevent unbounded growth from orphaned traces.
    """

    def __init__(self, redis_url: str, ttl_s: int = 300) -> None:
        import redis as redis_lib

        self._client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
        self._ttl_s = ttl_s

    def push(self, trace_id: str, span: Span) -> None:
        key = _REDIS_KEY_PREFIX + trace_id
        self._client.rpush(key, json.dumps(_span_to_dict(span)))
        self._client.expire(key, self._ttl_s)

    def pop_all(self, trace_id: str) -> list[Span]:
        key = _REDIS_KEY_PREFIX + trace_id
        pipe = self._client.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results = pipe.execute()
        raw_list: list[str] = results[0]
        return [_span_from_dict(json.loads(raw)) for raw in raw_list]

    def drain(self) -> list[Span]:
        all_spans: list[Span] = []
        cursor: int = 0
        while True:
            cursor, keys = self._client.scan(
                cursor=cursor, match=_REDIS_KEY_PREFIX + "*", count=100,
            )
            for key in keys:
                raw_list: list[str] = self._client.lrange(key, 0, -1)  # type: ignore[assignment]
                for raw in raw_list:
                    all_spans.append(_span_from_dict(json.loads(raw)))
                self._client.delete(key)
            if cursor == 0:
                break
        return all_spans

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()


# ---------------------------------------------------------------------------
# Span serialization for Redis
# ---------------------------------------------------------------------------


def _span_to_dict(span: Span) -> dict[str, Any]:
    """Serialize a Span to a JSON-safe dict."""
    return {
        "name": span.name,
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "kind": span.kind.value,
        "status": span.status.value,
        "attributes": span.attributes,
        "events": span.events,
        "start_time": span.start_time,
        "start_wall": span.start_wall.isoformat(),
        "end_time": span.end_time,
        "end_wall": span.end_wall.isoformat() if span.end_wall else None,
        "error_message": span.error_message,
    }


def _span_from_dict(d: dict[str, Any]) -> Span:
    """Deserialize a Span from a dict."""
    from datetime import datetime

    return Span(
        name=d["name"],
        trace_id=d["trace_id"],
        span_id=d["span_id"],
        parent_span_id=d.get("parent_span_id"),
        kind=SpanKind(d.get("kind", "internal")),
        status=SpanStatus(d.get("status", "unset")),
        attributes=d.get("attributes", {}),
        events=d.get("events", []),
        start_time=d["start_time"],
        start_wall=datetime.fromisoformat(d["start_wall"]).replace(tzinfo=UTC),
        end_time=d.get("end_time"),
        end_wall=(
            datetime.fromisoformat(d["end_wall"]).replace(tzinfo=UTC)
            if d.get("end_wall")
            else None
        ),
        error_message=d.get("error_message"),
    )


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class LangfuseSpanExporter:
    """Exports Drifter spans to Langfuse.

    Satisfies the ``SpanCollector`` protocol (``collect(span) -> None``).

    When ``redis_url`` is set in the config the exporter uses Redis to buffer
    child spans; otherwise it falls back to an in-memory dict.
    """

    def __init__(self, config: LangfuseConfig) -> None:
        self._config = config
        self._client: Any = None

        # Choose buffer backend based on config.
        if config.redis_url:
            self._buffer: SpanBuffer = RedisSpanBuffer(
                config.redis_url, ttl_s=config.buffer_ttl_s,
            )
            logger.info("Langfuse exporter using Redis buffer — %s", config.redis_url)
        else:
            self._buffer = InMemorySpanBuffer()

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the Langfuse client."""
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=self._config.public_key,
            secret_key=self._config.secret_key,
            host=self._config.host,
        )
        logger.info("Langfuse client connected — host=%s", self._config.host)

    def close(self) -> None:
        """Flush pending events and shut down."""
        if self._client is not None:
            # Flush any remaining buffered spans
            for span in self._buffer.drain():
                self._export_span(span)
            try:
                self._client.flush()
            except Exception:
                logger.exception("Error flushing Langfuse client")
            finally:
                with contextlib.suppress(Exception):
                    self._client.shutdown()
                self._client = None
        self._buffer.close()

    def health_check(self) -> bool:
        """Return True if the client is initialized."""
        return self._client is not None

    # -- SpanCollector protocol ----------------------------------------------

    def collect(self, span: Span) -> None:
        """Buffer child spans; flush the whole trace when the root arrives."""
        if self._client is None:
            return

        try:
            if span.parent_span_id is None:
                # Root span arrived — flush root first, then buffered children.
                children = self._buffer.pop_all(span.trace_id)
                self._export_root(span)
                for child in children:
                    self._export_span(child)
                self._client.flush()
            else:
                # Child span — buffer it.
                self._buffer.push(span.trace_id, span)
        except Exception:
            logger.exception("Failed to export span %r to Langfuse", span.name)

    # -- Internal ------------------------------------------------------------

    def _export_root(self, span: Span) -> None:
        """Export the root span and set the trace name."""
        trace_ctx: dict[str, str] = {"trace_id": span.trace_id}
        level = "ERROR" if span.status == SpanStatus.ERROR else "DEFAULT"
        metadata = _build_metadata(span)

        lf_span = self._client.start_as_current_span(
            trace_context=trace_ctx,
            name=span.name,
            input=_build_input(span),
            metadata=metadata,
            level=level,
            status_message=span.error_message,
        )
        with lf_span as s:
            s.update(output=_build_output(span))

    def _export_span(self, span: Span) -> None:
        """Export a child span or generation."""
        stage = span.attributes.get("pipeline.stage", "")
        trace_ctx: dict[str, str] = {"trace_id": span.trace_id}
        if span.parent_span_id:
            trace_ctx["parent_span_id"] = span.parent_span_id

        level = "ERROR" if span.status == SpanStatus.ERROR else "DEFAULT"

        if stage in _GENERATION_STAGES:
            self._create_generation(span, trace_ctx, level)
        else:
            self._create_span(span, trace_ctx, level)

    def _create_span(
        self, span: Span, trace_ctx: dict[str, str], level: str,
    ) -> None:
        """Create a Langfuse span observation."""
        metadata = _build_metadata(span)
        lf_span = self._client.start_span(
            trace_context=trace_ctx,
            name=span.name,
            input=_build_input(span),
            metadata=metadata,
            level=level,
            status_message=span.error_message,
        )
        lf_span.update(output=_build_output(span))
        lf_span.end()

    def _create_generation(
        self, span: Span, trace_ctx: dict[str, str], level: str,
    ) -> None:
        """Create a Langfuse generation observation for LLM calls."""
        attrs = span.attributes
        metadata = _build_metadata(span)

        # Use model_id directly (not generator_id which has a prefix)
        model = attrs.get("model_id") or attrs.get("generator_id", "unknown")
        prompt_tokens = attrs.get("prompt_tokens")
        completion_tokens = attrs.get("completion_tokens")

        usage_details: dict[str, int] | None = None
        if prompt_tokens is not None or completion_tokens is not None:
            usage_details = {}
            if prompt_tokens is not None:
                usage_details["input"] = int(prompt_tokens)
            if completion_tokens is not None and int(completion_tokens) > 0:
                usage_details["output"] = int(completion_tokens)

        lf_gen = self._client.start_observation(
            trace_context=trace_ctx,
            as_type="generation",
            name=span.name,
            model=str(model) if model else None,
            input=_build_input(span),
            metadata=metadata,
            level=level,
            status_message=span.error_message,
        )
        lf_gen.update(
            output=_build_output(span),
            usage_details=usage_details,
        )
        lf_gen.end()


# -- Helpers -----------------------------------------------------------------


def _build_metadata(span: Span) -> dict[str, Any]:
    """Build a metadata dict from span attributes, excluding known fields."""
    excluded = {
        "pipeline.stage", "outcome", "input_count", "output_count",
        "error_count", "model_id", "generator_id",
        "prompt_tokens", "completion_tokens",
    }
    return {
        k: v for k, v in span.attributes.items()
        if k not in excluded
    }


def _build_input(span: Span) -> dict[str, Any]:
    """Build the Langfuse input dict from span attributes."""
    attrs = span.attributes
    result: dict[str, Any] = {}
    if "pipeline.stage" in attrs:
        result["stage"] = attrs["pipeline.stage"]
    if "input_count" in attrs:
        result["input_count"] = attrs["input_count"]
    if "query" in attrs:
        result["query"] = attrs["query"]
    return result or {"stage": span.name}


def _build_output(span: Span) -> dict[str, Any]:
    """Build the Langfuse output dict from span attributes."""
    attrs = span.attributes
    result: dict[str, Any] = {}
    if "outcome" in attrs:
        result["outcome"] = attrs["outcome"]
    if "output_count" in attrs:
        result["output_count"] = attrs["output_count"]
    if "error_count" in attrs:
        result["error_count"] = attrs["error_count"]
    result["duration_ms"] = round(span.duration_ms, 2)
    if span.error_message:
        result["error"] = span.error_message
    return result
