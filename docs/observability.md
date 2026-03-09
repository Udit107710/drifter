# Observability Subsystem

Cross-cutting instrumentation for all pipeline stages. Every stage in the Drifter pipeline (ingest, parse, chunk, embed, index, retrieve, rerank, context build, generate) is traceable through a unified observability layer. The subsystem is OpenTelemetry-compatible in concept and field naming, but carries no hard dependency on the OTel SDK -- all abstractions are pure Python with zero external imports.

## Boundary

- **Input**: Called by every pipeline stage; receives `Tracer` + `ObservabilityContext`
- **Output**: Completed `Span` objects delivered to a `SpanCollector`; metrics recorded in `CounterMetric` / `HistogramMetric` singletons

The observability subsystem is a dependency of all other subsystems but depends on none of them. It must remain self-contained.

## Architecture

```
Query/Pipeline Stage
    │
    ▼ pipeline_span(tracer, ctx, stage)
    │
    ├── Span created with parent from ctx
    │   ├── Attributes set during execution
    │   ├── Events recorded (errors, milestones)
    │   └── Status set (OK on success, ERROR on exception)
    │
    ▼ Span ended → Collector.collect()
    │
    ├── InMemoryCollector (testing/debug)
    ├── NoOpCollector (disabled)
    ├── OtelSpanExporter (OpenTelemetry OTLP export)
    └── LangfuseSpanExporter (Langfuse trace export)
```

## Components

### Span (`spans.py`)

A mutable-until-ended record representing a unit of work. Fields are OTel-compatible:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Operation name (e.g. `"retrieval"`, `"reranking"`) |
| `trace_id` | `str` | Shared across all spans in a single pipeline run |
| `span_id` | `str` | Unique identifier for this span |
| `parent_span_id` | `str \| None` | Links child to parent for tree reconstruction |
| `kind` | `SpanKind` | INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER |
| `status` | `SpanStatus` | UNSET (in-flight), OK (success), ERROR (failure) |
| `attributes` | `dict[str, Any]` | Key-value metadata set during execution |
| `events` | `list[dict]` | Timestamped events (errors, milestones) |
| `start_time` / `end_time` | `float` | Monotonic clock for accurate duration |
| `start_wall` / `end_wall` | `datetime` | Wall-clock times for display and correlation |

Once `end()` is called, all mutating methods (`set_attribute`, `add_event`, `set_status`) become no-ops. The `duration_ms` property computes elapsed time from the monotonic timestamps. `to_dict()` exports the span as a plain dictionary suitable for JSON serialization or future OTel export.

### ObservabilityContext (`context.py`)

Carries trace identity and span nesting through the pipeline:

- `trace_id` -- generated as a UUID4 hex string (32 chars, OTel-compatible length)
- `span_stack` -- a list of span IDs representing the current nesting depth; the tracer pushes on entry and pops on exit
- `baggage` -- arbitrary key-value pairs propagated across stages (e.g. `query_id`, `experiment_id`)
- `current_span_id` -- property returning the top of the stack, used by the tracer to set `parent_span_id` on new spans

Helper functions `generate_trace_id()` and `generate_span_id()` produce IDs in OTel-compatible formats.

### SpanCollector Protocol + Implementations (`collector.py`)

A `@runtime_checkable` protocol with a single method: `collect(span: Span) -> None`.

**InMemoryCollector**: Stores completed spans in a list. Provides query methods (`find_by_name`, `find_by_trace`), a `count` property, and `clear()`. Ideal for testing and debugging.

**NoOpCollector**: Discards all spans. Used when observability is disabled or in benchmarks where collection overhead must be zero.

Two production collectors are implemented:

**OtelSpanExporter** (`libs/adapters/otel/exporter.py`): Exports spans via OTLP (gRPC or HTTP) to any OTel-compatible backend (Jaeger, Grafana Tempo, etc.). Configured via `DRIFTER_OTEL_ENDPOINT`.

**LangfuseSpanExporter** (`libs/adapters/langfuse/exporter.py`): Exports spans to Langfuse for LLM-focused observability. Features:
- Span buffering (in-memory or Redis-backed) — children are buffered until the root arrives, then flushed root-first
- `propagate_attributes(trace_name=...)` ensures traces keep the root span name
- Generation observations include model ID and token usage
- Wall-clock timestamps (`start_wall`, `end_wall`, `duration_ms`) are included in span metadata
- Configured via `DRIFTER_LANGFUSE_PUBLIC_KEY`, `_SECRET_KEY`, `_HOST`, `_REDIS_URL`

The bootstrap prefers Langfuse over OTel when both are configured. Additional collectors can be added by implementing the `SpanCollector` protocol.

### Tracer (`tracer.py`)

Context-managed span lifecycle:

1. `start_span(name, ctx)` creates a `Span` with `parent_span_id` from `ctx.current_span_id`
2. Pushes the new `span_id` onto `ctx.span_stack`
3. Yields the span for attribute/event recording
4. On exit (normal or exception): pops the stack, calls `span.end()`, submits to the collector
5. On exception: sets `SpanStatus.ERROR` and records an `"exception"` event with type and message before re-raising

The tracer also provides `create_context(trace_id=...)` as a convenience factory for `ObservabilityContext`.

### Pipeline Events (`events.py`)

Typed, frozen dataclasses for structured per-stage events. Each event carries `stage`, `trace_id`, and `attributes`, plus stage-specific fields:

| Event Class | Key Fields |
|-------------|------------|
| `IngestionEvent` | `source_count`, `new_sources`, `skipped_sources`, `error_count` |
| `ParsingEvent` | `document_id`, `block_count`, `parser_type`, `content_bytes` |
| `ChunkingEvent` | `document_id`, `input_block_count`, `output_chunk_count`, `strategy`, `avg_chunk_tokens` |
| `EmbeddingEvent` | `chunk_count`, `model_id`, `dimensions`, `batch_size` |
| `IndexingEvent` | `chunk_count`, `success_count`, `error_count`, `outcome` |
| `RetrievalEvent` | `mode`, `dense_count`, `lexical_count`, `fused_count`, `outcome` |
| `RerankingEvent` | `input_count`, `output_count`, `reranker_id`, `outcome` |
| `ContextBuildEvent` | `input_count`, `selected_count`, `excluded_count`, `tokens_used`, `token_budget`, `outcome` |
| `GenerationEvent` | `model_id`, `prompt_tokens`, `completion_tokens`, `citation_count`, `outcome` |

These events are intended for structured logging and event-driven observability. They complement spans (which track timing and parent-child relationships) with domain-specific payloads.

### Metrics (`metrics.py`)

Thread-safe metric abstractions:

**CounterMetric**: Monotonically increasing value. Methods: `increment(amount)`, `reset()`. Property: `value`.

**HistogramMetric**: Records a distribution of values (typically latencies). Methods: `record(value)`, `percentile(p)`, `reset()`. Properties: `count`, `sum`, `values`.

Three pre-defined module-level singletons are available:

| Metric | Type | Description |
|--------|------|-------------|
| `pipeline_latency` | `HistogramMetric` | Stage latency in milliseconds |
| `pipeline_errors` | `CounterMetric` | Error count across stages |
| `pipeline_throughput` | `CounterMetric` | Successful stage completions |

These are module-level singletons so that all pipeline stages contribute to the same aggregates. Call `reset()` before test runs to avoid cross-contamination.

### Stage Instruments (`stage_instruments.py`)

High-level helpers that combine spans and metrics:

**`pipeline_span(tracer, ctx, stage, *, kind, attributes)`**: A context manager that wraps `tracer.start_span()` and automatically:
- Sets `pipeline.stage` attribute on the span
- Increments `pipeline_throughput` on success
- Increments `pipeline_errors` on exception
- Records `pipeline_latency` with the span's `duration_ms` after completion

**`record_stage_result(span, *, outcome, input_count, output_count, error_count, extra)`**: Sets standard result attributes on a span. Call this inside the `pipeline_span` block before it exits. Records `outcome`, `input_count`, `output_count`, and any extra key-value pairs as span attributes.

## Instrumentation Points

Every stage in the pipeline should be instrumented with `pipeline_span()`:

| Stage | Span Name | Key Attributes |
|-------|-----------|----------------|
| Ingestion | `"ingestion"` | `source_count`, `new_sources`, `error_count` |
| Parsing | `"parsing"` | `document_id`, `parser_type`, `block_count` |
| Chunking | `"chunking"` | `document_id`, `strategy`, `output_chunk_count` |
| Embedding | `"embedding"` | `model_id`, `chunk_count`, `dimensions` |
| Indexing | `"indexing"` | `chunk_count`, `success_count`, `outcome` |
| Retrieval | `"retrieval"` | `mode`, `dense_count`, `lexical_count`, `fused_count` |
| Reranking | `"reranking"` | `reranker_id`, `input_count`, `output_count` |
| Context Build | `"context_build"` | `selected_count`, `tokens_used`, `token_budget` |
| Generation | `"generation"` | `model_id`, `prompt_tokens`, `completion_tokens`, `citation_count` |

## Trace ID Propagation

A single `ObservabilityContext` instance is created at the entry point of a pipeline run (e.g. an API request or batch job). Its `trace_id` is shared by every span created during that run:

1. The entry point creates `ctx = ObservabilityContext()` (auto-generates `trace_id`) or `tracer.create_context(trace_id="...")` with an externally provided ID
2. `ctx` is passed to each stage function alongside the tracer
3. Each `pipeline_span(tracer, ctx, stage)` call creates a child span whose `parent_span_id` comes from `ctx.current_span_id` (the top of the span stack)
4. Nested calls (e.g. retrieval invoking multiple store queries) produce a tree of spans, all sharing the same `trace_id`
5. After collection, `InMemoryCollector.find_by_trace(trace_id)` retrieves the complete span tree for a single pipeline run

The `baggage` dict on `ObservabilityContext` can carry additional identifiers (experiment ID, user ID, query ID) that stages can read without coupling to each other.

## Usage

```python
import time

from libs.observability import (
    InMemoryCollector,
    ObservabilityContext,
    Tracer,
    pipeline_span,
    record_stage_result,
)

# Set up
collector = InMemoryCollector()
tracer = Tracer(collector=collector, service_name="drifter")
ctx = ObservabilityContext()

# Instrument a multi-stage pipeline
with pipeline_span(tracer, ctx, "retrieval", attributes={"mode": "hybrid"}) as span:
    time.sleep(0.01)  # simulate work
    record_stage_result(span, outcome="SUCCESS", input_count=1, output_count=25)

with pipeline_span(tracer, ctx, "reranking") as span:
    time.sleep(0.005)
    record_stage_result(span, outcome="SUCCESS", input_count=25, output_count=10)

with pipeline_span(tracer, ctx, "context_build") as span:
    time.sleep(0.002)
    record_stage_result(
        span,
        outcome="SUCCESS",
        input_count=10,
        output_count=5,
        extra={"tokens_used": 1800, "token_budget": 4096},
    )

with pipeline_span(tracer, ctx, "generation") as span:
    time.sleep(0.02)
    span.add_event("llm_call_start", {"model": "mock-v1"})
    record_stage_result(
        span,
        outcome="SUCCESS",
        input_count=5,
        output_count=1,
        extra={"model_id": "mock-v1", "citation_count": 3},
    )

# Inspect results
for s in collector.spans:
    print(f"{s.name}: {s.duration_ms:.1f}ms  status={s.status.value}")
```

## Metrics

The pre-defined metrics aggregate across all pipeline runs:

- **`pipeline_latency`** (`HistogramMetric`): Records the `duration_ms` of every `pipeline_span` invocation. Use `percentile(50)` for median, `percentile(99)` for tail latency. Histogram values can be exported to OTel-compatible backends for bucketed analysis.

- **`pipeline_errors`** (`CounterMetric`): Incremented each time a `pipeline_span` catches an exception. A rising error rate signals degradation.

- **`pipeline_throughput`** (`CounterMetric`): Incremented on each successful stage completion. Useful for tracking processing volume.

All three metrics are thread-safe and support `reset()` for test isolation.

## OpenTelemetry Integration

Every abstraction in this subsystem maps directly to an OTel concept:

| Drifter Abstraction | OTel Equivalent |
|---------------------|-----------------|
| `Span` | `opentelemetry.trace.Span` |
| `SpanStatus` | `opentelemetry.trace.StatusCode` |
| `SpanKind` | `opentelemetry.trace.SpanKind` |
| `ObservabilityContext` | `opentelemetry.context.Context` + `SpanContext` |
| `Tracer` | `opentelemetry.trace.Tracer` |
| `SpanCollector` | `opentelemetry.sdk.trace.export.SpanExporter` |
| `CounterMetric` | `opentelemetry.metrics.Counter` |
| `HistogramMetric` | `opentelemetry.metrics.Histogram` |
| `Span.to_dict()` | OTel protobuf/JSON span export format |

Both `OtelSpanExporter` and `LangfuseSpanExporter` implement the `SpanCollector` protocol. The bootstrap selects the collector based on which env vars are set (Langfuse preferred when both are configured). No pipeline code changes are required — only the collector wiring in `orchestrators/bootstrap.py`.

## Structured Logging

All core services use Python's `logging` module with `logger = logging.getLogger(__name__)`:

| Level | Usage |
|-------|-------|
| `DEBUG` | Stage entry with input parameters |
| `INFO` | Stage completion with output counts and latency; pipeline entry/exit |
| `WARNING` | Degraded operation (reranking fallback, partial store failure, empty input) |
| `ERROR` | Stage failure with exception details |

Enable verbose logging with `rag -v` or by setting `DRIFTER_LOG_LEVEL=DEBUG`.

## Skills Applied

This subsystem was built following these skills:
- `reliability/production_observability` -- structured spans, metrics, and event types for every pipeline stage
- `reliability/slo_sli_thinking` -- latency histograms with percentile support for SLI measurement
- `rag/rag_trace_analysis` -- trace_id propagation, parent-child span trees, per-stage attribute recording
- `reliability/failure_mode_analysis` -- error status on spans, exception events with type and message, error counters
