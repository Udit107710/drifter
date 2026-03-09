"""Langfuse span exporter adapter.

Converts Drifter spans into Langfuse traces, spans, and generations.
The ``generation`` pipeline stage is exported as a Langfuse generation
observation with model and token usage metadata; all other stages are
exported as plain Langfuse spans.

Implements the ``SpanCollector`` protocol.

Compatible with Langfuse SDK v3 (OTel-based API).
"""

from __future__ import annotations

import logging
from typing import Any

from libs.adapters.config import LangfuseConfig
from libs.observability.spans import Span, SpanStatus

logger = logging.getLogger(__name__)

# Pipeline stages that represent an LLM call.
_GENERATION_STAGES = frozenset({"generation"})


class LangfuseSpanExporter:
    """Exports Drifter spans to Langfuse.

    Satisfies the ``SpanCollector`` protocol (``collect(span) -> None``).

    Design:
    - Uses the Langfuse SDK v3 OTel-based API.
    - Each span is exported with a ``trace_context`` linking it to the
      correct trace_id and parent span.
    - The "generation" pipeline stage is exported as a Langfuse *generation*
      with model, token usage, and input/output metadata.
    - All other stages are exported as Langfuse *spans*.
    """

    def __init__(self, config: LangfuseConfig) -> None:
        self._config = config
        self._client: Any = None

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
            try:
                self._client.flush()
            except Exception:
                logger.exception("Error flushing Langfuse client")
            finally:
                try:
                    self._client.shutdown()
                except Exception:
                    pass
                self._client = None

    def health_check(self) -> bool:
        """Return True if the client is initialized."""
        return self._client is not None

    # -- SpanCollector protocol ----------------------------------------------

    def collect(self, span: Span) -> None:
        """Convert a Drifter Span to a Langfuse observation and submit it."""
        if self._client is None:
            return

        try:
            self._do_collect(span)
        except Exception:
            logger.exception("Failed to export span %r to Langfuse", span.name)

    # -- Internal ------------------------------------------------------------

    def _do_collect(self, span: Span) -> None:
        """Core collection logic."""
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
            output=_build_output(span),
            metadata=metadata,
            level=level,
            status_message=span.error_message,
        )
        lf_span.end()

    def _create_generation(
        self, span: Span, trace_ctx: dict[str, str], level: str,
    ) -> None:
        """Create a Langfuse generation observation for LLM calls."""
        attrs = span.attributes
        metadata = _build_metadata(span)

        model = attrs.get("model_id", attrs.get("generator_id", "unknown"))
        prompt_tokens = attrs.get("prompt_tokens")
        completion_tokens = attrs.get("completion_tokens")

        usage_details: dict[str, int] | None = None
        if prompt_tokens is not None or completion_tokens is not None:
            usage_details = {}
            if prompt_tokens is not None:
                usage_details["input"] = int(prompt_tokens)
            if completion_tokens is not None:
                usage_details["output"] = int(completion_tokens)

        lf_gen = self._client.start_observation(
            trace_context=trace_ctx,
            as_type="generation",
            name=span.name,
            model=str(model) if model else None,
            input=_build_input(span),
            output=_build_output(span),
            usage_details=usage_details,
            metadata=metadata,
            level=level,
            status_message=span.error_message,
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
