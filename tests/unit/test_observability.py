"""Tests for the observability subsystem."""

from __future__ import annotations

import pytest

from libs.observability.collector import (
    InMemoryCollector,
    NoOpCollector,
    SpanCollector,
)
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
from libs.observability.spans import Span, SpanStatus
from libs.observability.stage_instruments import pipeline_span, record_stage_result
from libs.observability.tracer import Tracer

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global_metrics():
    """Reset module-level singleton metrics before each test."""
    pipeline_latency.reset()
    pipeline_errors.reset()
    pipeline_throughput.reset()
    yield


def _make_span(
    name: str = "test-span",
    trace_id: str = "a" * 32,
    span_id: str = "b" * 16,
    **kwargs,
) -> Span:
    return Span(name=name, trace_id=trace_id, span_id=span_id, **kwargs)


# ── TestSpan ────────────────────────────────────────────────────────


class TestSpan:
    def test_create_and_end(self):
        span = _make_span()
        assert span.duration_ms == 0.0
        assert not span.is_ended
        span.end()
        assert span.is_ended
        assert span.duration_ms >= 0.0

    def test_set_attribute(self):
        span = _make_span()
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)
        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42

    def test_set_attribute_after_end_ignored(self):
        span = _make_span()
        span.set_attribute("before", 1)
        span.end()
        span.set_attribute("after", 2)
        assert "before" in span.attributes
        assert "after" not in span.attributes

    def test_add_event(self):
        span = _make_span()
        span.add_event("my-event", {"detail": "info"})
        assert len(span.events) == 1
        event = span.events[0]
        assert event["name"] == "my-event"
        assert event["attributes"]["detail"] == "info"
        assert "timestamp" in event

    def test_add_event_after_end_ignored(self):
        span = _make_span()
        span.end()
        span.add_event("late-event")
        assert len(span.events) == 0

    def test_set_status_error(self):
        span = _make_span()
        span.set_status(SpanStatus.ERROR, "something broke")
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "something broke"

    def test_set_status_after_end_ignored(self):
        span = _make_span()
        span.end()
        span.set_status(SpanStatus.ERROR, "too late")
        assert span.status == SpanStatus.OK  # end() sets UNSET -> OK

    def test_end_sets_ok_if_unset(self):
        span = _make_span()
        assert span.status == SpanStatus.UNSET
        span.end()
        assert span.status == SpanStatus.OK

    def test_end_preserves_error_status(self):
        span = _make_span()
        span.set_status(SpanStatus.ERROR, "fail")
        span.end()
        assert span.status == SpanStatus.ERROR

    def test_double_end_idempotent(self):
        span = _make_span()
        span.end()
        first_end_time = span.end_time
        first_end_wall = span.end_wall
        span.end()
        assert span.end_time == first_end_time
        assert span.end_wall == first_end_wall

    def test_to_dict(self):
        span = _make_span(parent_span_id="c" * 16)
        span.set_attribute("foo", "bar")
        span.add_event("evt")
        span.end()
        d = span.to_dict()
        expected_keys = {
            "name",
            "trace_id",
            "span_id",
            "parent_span_id",
            "kind",
            "status",
            "attributes",
            "events",
            "start_wall",
            "end_wall",
            "duration_ms",
            "error_message",
        }
        assert set(d.keys()) == expected_keys
        assert d["name"] == "test-span"
        assert d["trace_id"] == "a" * 32
        assert d["span_id"] == "b" * 16
        assert d["parent_span_id"] == "c" * 16
        assert d["kind"] == "internal"
        assert d["status"] == "ok"
        assert d["attributes"]["foo"] == "bar"
        assert len(d["events"]) == 1
        assert d["duration_ms"] >= 0.0
        assert d["end_wall"] is not None
        assert d["error_message"] is None


# ── TestObservabilityContext ────────────────────────────────────────


class TestObservabilityContext:
    def test_generate_trace_id_length(self):
        tid = generate_trace_id()
        assert len(tid) == 32
        # Must be valid hex
        int(tid, 16)

    def test_generate_span_id_length(self):
        sid = generate_span_id()
        assert len(sid) == 16
        int(sid, 16)

    def test_generate_ids_unique(self):
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_span_stack_push_pop(self):
        ctx = ObservabilityContext(trace_id="a" * 32)
        ctx.push_span("span-1")
        ctx.push_span("span-2")
        assert ctx.current_span_id == "span-2"
        assert ctx.pop_span() == "span-2"
        assert ctx.current_span_id == "span-1"
        assert ctx.pop_span() == "span-1"
        assert ctx.current_span_id is None

    def test_current_span_id_empty(self):
        ctx = ObservabilityContext()
        assert ctx.current_span_id is None

    def test_pop_empty_returns_none(self):
        ctx = ObservabilityContext()
        assert ctx.pop_span() is None

    def test_baggage(self):
        ctx = ObservabilityContext(baggage={"user_id": "u123", "session": "s456"})
        assert ctx.baggage["user_id"] == "u123"
        assert ctx.baggage["session"] == "s456"

    def test_default_trace_id_generated(self):
        ctx = ObservabilityContext()
        assert len(ctx.trace_id) == 32


# ── TestCollectors ──────────────────────────────────────────────────


class TestCollectors:
    def test_in_memory_collector_protocol(self):
        collector = InMemoryCollector()
        assert isinstance(collector, SpanCollector)

    def test_no_op_collector_protocol(self):
        collector = NoOpCollector()
        assert isinstance(collector, SpanCollector)

    def test_in_memory_collect_and_query(self):
        collector = InMemoryCollector()
        span = _make_span()
        span.end()
        collector.collect(span)
        assert collector.count == 1
        assert collector.spans[0].name == "test-span"

    def test_find_by_name(self):
        collector = InMemoryCollector()
        s1 = _make_span(name="retrieval")
        s2 = _make_span(name="generation")
        s3 = _make_span(name="retrieval")
        for s in (s1, s2, s3):
            s.end()
            collector.collect(s)
        found = collector.find_by_name("retrieval")
        assert len(found) == 2
        assert all(s.name == "retrieval" for s in found)

    def test_find_by_trace(self):
        collector = InMemoryCollector()
        trace_a = "a" * 32
        trace_b = "b" * 32
        s1 = _make_span(trace_id=trace_a)
        s2 = _make_span(trace_id=trace_b)
        s3 = _make_span(trace_id=trace_a)
        for s in (s1, s2, s3):
            s.end()
            collector.collect(s)
        found = collector.find_by_trace(trace_a)
        assert len(found) == 2
        assert all(s.trace_id == trace_a for s in found)

    def test_clear(self):
        collector = InMemoryCollector()
        collector.collect(_make_span())
        collector.collect(_make_span())
        assert collector.count == 2
        collector.clear()
        assert collector.count == 0
        assert collector.spans == []

    def test_no_op_collector_discards(self):
        collector = NoOpCollector()
        span = _make_span()
        collector.collect(span)
        # NoOpCollector has no state to inspect; just verify no exception

    def test_spans_returns_copy(self):
        collector = InMemoryCollector()
        collector.collect(_make_span())
        spans = collector.spans
        spans.clear()
        assert collector.count == 1  # internal list unaffected


# ── TestTracer ──────────────────────────────────────────────────────


class TestTracer:
    def test_span_auto_ended_and_collected(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        with tracer.start_span("test-op", ctx) as span:
            assert not span.is_ended

        assert span.is_ended
        assert span.status == SpanStatus.OK
        assert collector.count == 1
        assert collector.spans[0] is span

    def test_parent_child_nesting(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        with tracer.start_span("parent", ctx) as parent_span:  # noqa: SIM117
            with tracer.start_span("child", ctx) as child_span:
                pass

        assert child_span.parent_span_id == parent_span.span_id
        assert parent_span.parent_span_id is None
        assert collector.count == 2
        # Context stack should be empty after both exit
        assert ctx.current_span_id is None

    def test_exception_sets_error_status(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        with pytest.raises(ValueError, match="boom"), tracer.start_span("failing", ctx) as span:
            raise ValueError("boom")

        assert span.status == SpanStatus.ERROR
        assert span.error_message == "boom"
        assert len(span.events) == 1
        exc_event = span.events[0]
        assert exc_event["name"] == "exception"
        assert exc_event["attributes"]["exception.type"] == "ValueError"
        assert exc_event["attributes"]["exception.message"] == "boom"
        assert span.is_ended
        assert collector.count == 1

    def test_service_name_attribute(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector, service_name="my-service")
        ctx = ObservabilityContext()
        assert tracer.service_name == "my-service"

        with tracer.start_span("op", ctx) as span:
            pass

        assert span.attributes["service.name"] == "my-service"

    def test_create_context(self):
        tracer = Tracer()
        ctx = tracer.create_context(trace_id="abc123")
        assert ctx.trace_id == "abc123"
        assert ctx.span_stack == []
        assert ctx.baggage == {}

    def test_create_context_auto_id(self):
        tracer = Tracer()
        ctx = tracer.create_context()
        assert len(ctx.trace_id) == 32

    def test_default_service_name(self):
        tracer = Tracer()
        assert tracer.service_name == "drifter"

    def test_default_collector_is_noop(self):
        tracer = Tracer()
        ctx = ObservabilityContext()
        # Should not raise even without a collector
        with tracer.start_span("noop-span", ctx) as span:
            span.set_attribute("key", "value")
        assert span.is_ended


# ── TestEvents ──────────────────────────────────────────────────────


class TestEvents:
    def test_pipeline_event_base(self):
        evt = PipelineEvent(stage="test", trace_id="t" * 32)
        assert evt.stage == "test"
        assert evt.trace_id == "t" * 32
        assert evt.attributes == {}

    def test_pipeline_event_with_attributes(self):
        evt = PipelineEvent(
            stage="test",
            trace_id="t" * 32,
            attributes={"key": "value"},
        )
        assert evt.attributes["key"] == "value"

    def test_all_stage_events(self):
        trace_id = "t" * 32
        events = [
            IngestionEvent(stage="ingestion", trace_id=trace_id, source_count=5, new_sources=3),
            ParsingEvent(stage="parsing", trace_id=trace_id, document_id="d1", block_count=10),
            ChunkingEvent(
                stage="chunking", trace_id=trace_id,
                output_chunk_count=20, strategy="fixed",
            ),
            EmbeddingEvent(
                stage="embedding", trace_id=trace_id,
                chunk_count=20, model_id="e5",
            ),
            IndexingEvent(
                stage="indexing", trace_id=trace_id,
                chunk_count=20, success_count=18,
            ),
            RetrievalEvent(
                stage="retrieval", trace_id=trace_id,
                mode="hybrid", fused_count=10,
            ),
            RerankingEvent(
                stage="reranking", trace_id=trace_id,
                input_count=10, output_count=5,
            ),
            ContextBuildEvent(
                stage="context_build", trace_id=trace_id,
                selected_count=3, tokens_used=500,
            ),
            GenerationEvent(
                stage="generation", trace_id=trace_id,
                model_id="gpt", completion_tokens=100,
            ),
        ]
        assert len(events) == 9
        for evt in events:
            assert isinstance(evt, PipelineEvent)
            assert evt.trace_id == trace_id

        # Verify stage-specific field values
        assert events[0].source_count == 5
        assert events[1].block_count == 10
        assert events[2].strategy == "fixed"
        assert events[3].model_id == "e5"
        assert events[4].success_count == 18
        assert events[5].mode == "hybrid"
        assert events[6].output_count == 5
        assert events[7].tokens_used == 500
        assert events[8].completion_tokens == 100

    def test_event_immutability(self):
        evt = IngestionEvent(stage="ingestion", trace_id="t" * 32, source_count=5)
        with pytest.raises(AttributeError):
            evt.source_count = 10  # type: ignore[misc]
        with pytest.raises(AttributeError):
            evt.stage = "other"  # type: ignore[misc]

    def test_event_defaults(self):
        evt = IngestionEvent(stage="ingestion", trace_id="t" * 32)
        assert evt.source_count == 0
        assert evt.new_sources == 0
        assert evt.skipped_sources == 0
        assert evt.error_count == 0


# ── TestMetrics ─────────────────────────────────────────────────────


class TestMetrics:
    def test_counter_increment(self):
        counter = CounterMetric(name="test.counter")
        assert counter.value == 0.0
        counter.increment()
        assert counter.value == 1.0
        counter.increment(5.0)
        assert counter.value == 6.0

    def test_counter_reset(self):
        counter = CounterMetric(name="test.counter")
        counter.increment(10.0)
        counter.reset()
        assert counter.value == 0.0

    def test_histogram_record(self):
        hist = HistogramMetric(name="test.hist", unit="ms")
        hist.record(10.0)
        hist.record(20.0)
        hist.record(30.0)
        assert hist.count == 3
        assert hist.sum == 60.0
        assert hist.values == [10.0, 20.0, 30.0]

    def test_histogram_percentile(self):
        hist = HistogramMetric(name="test.hist")
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            hist.record(v)
        p50 = hist.percentile(50)
        p90 = hist.percentile(90)
        p99 = hist.percentile(99)
        # p50 should be around 5-6, p90 around 9-10
        assert 4.0 <= p50 <= 6.0
        assert 9.0 <= p90 <= 10.0
        assert p99 == 10.0

    def test_histogram_empty_percentile(self):
        hist = HistogramMetric(name="test.hist")
        assert hist.percentile(50) == 0.0
        assert hist.percentile(99) == 0.0

    def test_histogram_reset(self):
        hist = HistogramMetric(name="test.hist")
        hist.record(1.0)
        hist.record(2.0)
        hist.reset()
        assert hist.count == 0
        assert hist.sum == 0.0
        assert hist.values == []

    def test_counter_default_description(self):
        counter = CounterMetric(name="c")
        assert counter.description == ""

    def test_histogram_default_unit(self):
        hist = HistogramMetric(name="h")
        assert hist.unit == "ms"

    def test_predefined_metrics_exist(self):
        assert pipeline_latency.name == "drifter.pipeline.stage.latency"
        assert pipeline_errors.name == "drifter.pipeline.stage.errors"
        assert pipeline_throughput.name == "drifter.pipeline.throughput"


# ── TestStageInstruments ────────────────────────────────────────────


class TestStageInstruments:
    def test_pipeline_span_success(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        with pipeline_span(tracer, ctx, "retrieval") as span:
            span.set_attribute("candidate_count", 42)

        # Span collected
        assert collector.count == 1
        collected = collector.spans[0]
        assert collected.name == "retrieval"
        assert collected.status == SpanStatus.OK
        assert collected.attributes["pipeline.stage"] == "retrieval"
        assert collected.attributes["candidate_count"] == 42

        # Metrics updated
        assert pipeline_throughput.value == 1.0
        assert pipeline_latency.count == 1
        assert pipeline_latency.values[0] >= 0.0
        assert pipeline_errors.value == 0.0

    def test_pipeline_span_error(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        with (
            pytest.raises(RuntimeError, match="stage failed"),
            pipeline_span(tracer, ctx, "indexing"),
        ):
            raise RuntimeError("stage failed")

        # Error counter incremented
        assert pipeline_errors.value == 1.0
        # Throughput not incremented on error
        assert pipeline_throughput.value == 0.0
        # Span still collected with ERROR status
        assert collector.count == 1
        assert collector.spans[0].status == SpanStatus.ERROR
        # Latency NOT recorded on error (exception exits before latency recording)
        assert pipeline_latency.count == 0

    def test_record_stage_result(self):
        span = _make_span()
        record_stage_result(
            span,
            outcome="success",
            input_count=100,
            output_count=50,
            error_count=2,
            extra={"custom_key": "custom_val"},
        )
        assert span.attributes["outcome"] == "success"
        assert span.attributes["input_count"] == 100
        assert span.attributes["output_count"] == 50
        assert span.attributes["error_count"] == 2
        assert span.attributes["custom_key"] == "custom_val"

    def test_record_stage_result_no_errors(self):
        span = _make_span()
        record_stage_result(span, outcome="success", input_count=10, output_count=10)
        assert "error_count" not in span.attributes
        assert span.attributes["outcome"] == "success"

    def test_record_stage_result_no_extra(self):
        span = _make_span()
        record_stage_result(span, outcome="partial")
        assert span.attributes["outcome"] == "partial"
        assert span.attributes["input_count"] == 0
        assert span.attributes["output_count"] == 0

    def test_nested_pipeline_spans(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        with pipeline_span(tracer, ctx, "retrieval") as outer_span:  # noqa: SIM117
            with pipeline_span(tracer, ctx, "dense_retrieval") as inner_span:
                inner_span.set_attribute("mode", "dense")

        assert collector.count == 2
        # Inner span should have outer as parent
        assert inner_span.parent_span_id == outer_span.span_id
        assert outer_span.parent_span_id is None

        # Both throughput increments recorded
        assert pipeline_throughput.value == 2.0
        # Both latencies recorded
        assert pipeline_latency.count == 2

    def test_multiple_pipeline_spans_accumulate_metrics(self):
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = ObservabilityContext(trace_id="t" * 32)

        for stage in ["parsing", "chunking", "embedding"]:
            with pipeline_span(tracer, ctx, stage):
                pass

        assert pipeline_throughput.value == 3.0
        assert pipeline_latency.count == 3
        assert collector.count == 3


# ── TestIntegration ─────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline_trace(self):
        """Simulate a multi-stage pipeline trace with nested spans.

        Verifies parent-child chain, all spans collected, correct trace IDs,
        and metric accumulation across stages.
        """
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector, service_name="drifter-test")
        ctx = tracer.create_context(trace_id="integration" + "0" * 21)  # 32 chars

        # Stage 1: retrieval (outer)
        attrs = {"mode": "hybrid"}
        with pipeline_span(tracer, ctx, "retrieval", attributes=attrs) as retrieval_span:
            record_stage_result(retrieval_span, outcome="success", input_count=1, output_count=50)

            # Stage 2: dense retrieval (child of retrieval)
            with pipeline_span(tracer, ctx, "dense_retrieval") as dense_span:
                record_stage_result(dense_span, outcome="success", input_count=1, output_count=30)

            # Stage 3: lexical retrieval (child of retrieval)
            with pipeline_span(tracer, ctx, "lexical_retrieval") as lexical_span:
                record_stage_result(lexical_span, outcome="success", input_count=1, output_count=20)

        # All 3 spans collected
        assert collector.count == 3

        # All spans share the same trace_id
        trace_id = "integration" + "0" * 21
        trace_spans = collector.find_by_trace(trace_id)
        assert len(trace_spans) == 3

        # All spans have service.name
        for s in trace_spans:
            assert s.attributes["service.name"] == "drifter-test"
            assert s.is_ended
            assert s.status == SpanStatus.OK

        # Parent-child chain
        assert retrieval_span.parent_span_id is None
        assert dense_span.parent_span_id == retrieval_span.span_id
        assert lexical_span.parent_span_id == retrieval_span.span_id

        # Span names
        names = [s.name for s in collector.spans]
        assert "dense_retrieval" in names
        assert "lexical_retrieval" in names
        assert "retrieval" in names

        # Metrics
        assert pipeline_throughput.value == 3.0
        assert pipeline_latency.count == 3
        assert pipeline_errors.value == 0.0

        # Context stack empty after all spans exit
        assert ctx.current_span_id is None

        # Verify find_by_name
        assert len(collector.find_by_name("retrieval")) == 1
        assert len(collector.find_by_name("dense_retrieval")) == 1

    def test_mixed_success_and_error_trace(self):
        """Pipeline where one stage fails after others succeed."""
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        ctx = tracer.create_context()

        with pipeline_span(tracer, ctx, "parsing") as parsing_span:
            record_stage_result(parsing_span, outcome="success", output_count=10)

        with pytest.raises(ValueError), pipeline_span(tracer, ctx, "chunking") as chunking_span:
            raise ValueError("chunk size too small")

        assert collector.count == 2
        assert parsing_span.status == SpanStatus.OK
        assert chunking_span.status == SpanStatus.ERROR
        assert pipeline_throughput.value == 1.0  # only parsing succeeded
        assert pipeline_errors.value == 1.0
        assert pipeline_latency.count == 1  # only parsing recorded latency (error skips recording)
        assert ctx.current_span_id is None
