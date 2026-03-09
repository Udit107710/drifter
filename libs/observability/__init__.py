"""Observability subsystem.

Responsibilities:
- Provide OpenTelemetry-compatible span helpers for every pipeline stage
- Emit structured metrics (latency, throughput, error rates)
- Structured event types for each pipeline stage
- Cross-cutting: used by all other subsystems

Every stage (ingest, parse, chunk, embed, index, retrieve, rerank, pack, generate)
must be traceable.
"""

from libs.observability.collector import InMemoryCollector, NoOpCollector, SpanCollector
from libs.observability.context import (
    ObservabilityContext,
    generate_span_id,
    generate_trace_id,
)
from libs.observability.events import (
    ChunkingEvent,
    ContextBuildEvent,
    EmbeddingEvent,
    GenerationEvent,
    IndexingEvent,
    IngestionEvent,
    ParsingEvent,
    PipelineEvent,
    RerankingEvent,
    RetrievalEvent,
)
from libs.observability.metrics import (
    CounterMetric,
    HistogramMetric,
    pipeline_errors,
    pipeline_latency,
    pipeline_throughput,
)
from libs.observability.spans import Span, SpanKind, SpanStatus
from libs.observability.stage_instruments import pipeline_span, record_stage_result
from libs.observability.tracer import Tracer

__all__ = [
    "ChunkingEvent",
    "ContextBuildEvent",
    "CounterMetric",
    "EmbeddingEvent",
    "GenerationEvent",
    "HistogramMetric",
    "InMemoryCollector",
    "IndexingEvent",
    "IngestionEvent",
    "NoOpCollector",
    "ObservabilityContext",
    "ParsingEvent",
    "PipelineEvent",
    "RerankingEvent",
    "RetrievalEvent",
    "Span",
    "SpanCollector",
    "SpanKind",
    "SpanStatus",
    "Tracer",
    "generate_span_id",
    "generate_trace_id",
    "pipeline_errors",
    "pipeline_latency",
    "pipeline_span",
    "pipeline_throughput",
    "record_stage_result",
]
