# libs/observability/

Cross-cutting OpenTelemetry-compatible instrumentation for all pipeline stages.

## Boundary

- **Cross-cutting concern:** Used by all other subsystems
- **Provides:** Span creation, metric recording, structured events
- **Rule:** Every pipeline stage must be traceable.

## Core Components

### Tracer (`tracer.py`)

```python
class Tracer:
    def __init__(self, collector: SpanCollector | None = None, service_name: str = "drifter") -> None: ...
    def start_span(self, name: str, ctx: ObservabilityContext, ...) -> Generator[Span, None, None]: ...
    def create_context(self, trace_id: str | None = None) -> ObservabilityContext: ...
```

Context manager that creates spans, manages the span stack, handles exceptions, and collects completed spans.

### ObservabilityContext (`context.py`)

```python
@dataclass
class ObservabilityContext:
    trace_id: str          # UUID4 hex (auto-generated or provided)
    span_stack: list[str]  # Tracks nesting for parent_span_id
    baggage: dict[str, str] # Arbitrary key-value propagation
```

Flows from CLI → orchestrator → every service → every span. The trace_id links all spans in a request.

### Span (`spans.py`)

```python
@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    kind: SpanKind          # INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER
    status: SpanStatus      # OK, ERROR, UNSET
    attributes: dict
    events: list[dict]
    start_time: float
    duration_ms: float | None
```

## Stage Instrumentation (`stage_instruments.py`)

```python
def pipeline_span(tracer, ctx, stage, ...) -> Generator[Span, None, None]: ...
def record_stage_result(span, *, outcome, input_count, output_count, error_count, extra): ...
```

Wraps a pipeline stage in a span and auto-records `pipeline_throughput`, `pipeline_errors`, and `pipeline_latency` metrics.

## Collectors

| Collector | Purpose |
|-----------|---------|
| `SpanCollector` | Protocol: receives completed spans |
| `InMemoryCollector` | Stores spans in memory (testing/debugging). Supports `find_by_name`, `find_by_trace`. |
| `NoOpCollector` | Discards all spans (when observability is disabled) |
| `OtelSpanExporter` | Exports to OpenTelemetry (in `libs/adapters/otel/`) |

## Metrics (`metrics.py`)

Pre-defined metric instruments:

| Metric | Type | Purpose |
|--------|------|---------|
| `pipeline_throughput` | Counter | Successful stage completions |
| `pipeline_errors` | Counter | Stage failures |
| `pipeline_latency` | Histogram | Per-stage latency in ms |

## Events (`events.py`)

Typed event classes for each pipeline stage:

| Event | Stage |
|-------|-------|
| `IngestionEvent` | Ingestion |
| `ParsingEvent` | Parsing |
| `ChunkingEvent` | Chunking |
| `EmbeddingEvent` | Embedding |
| `IndexingEvent` | Indexing |
| `RetrievalEvent` | Retrieval |
| `RerankingEvent` | Reranking |
| `ContextBuildEvent` | Context building |
| `GenerationEvent` | Generation |

## Testing

Uses `InMemoryCollector` to capture and assert on spans. Tests verify span names, trace ID propagation, and parent-child relationships.
