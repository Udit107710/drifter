"""Example: observability instrumentation across a multi-stage pipeline trace."""

from __future__ import annotations

import time

from libs.observability import (
    InMemoryCollector,
    ObservabilityContext,
    Tracer,
    pipeline_errors,
    pipeline_latency,
    pipeline_span,
    pipeline_throughput,
    record_stage_result,
)


def simulate_retrieval(tracer: Tracer, ctx: ObservabilityContext) -> int:
    """Simulate retrieval stage returning candidate count."""
    with pipeline_span(tracer, ctx, "retrieval", attributes={"mode": "hybrid"}) as span:
        time.sleep(0.015)
        candidate_count = 25
        span.set_attribute("dense_count", 20)
        span.set_attribute("lexical_count", 18)
        span.set_attribute("fused_count", candidate_count)
        span.add_event("fusion_complete", {"strategy": "rrf", "k": 60})
        record_stage_result(
            span,
            outcome="SUCCESS",
            input_count=1,
            output_count=candidate_count,
            extra={"retrieval_method": "hybrid"},
        )
    return candidate_count


def simulate_reranking(
    tracer: Tracer, ctx: ObservabilityContext, input_count: int
) -> int:
    """Simulate reranking stage."""
    with pipeline_span(tracer, ctx, "reranking") as span:
        time.sleep(0.008)
        output_count = min(input_count, 10)
        span.set_attribute("reranker_id", "feature-based-v1")
        record_stage_result(
            span,
            outcome="SUCCESS",
            input_count=input_count,
            output_count=output_count,
            extra={"reranker_id": "feature-based-v1", "truncated": input_count - output_count},
        )
    return output_count


def simulate_context_build(
    tracer: Tracer, ctx: ObservabilityContext, input_count: int
) -> int:
    """Simulate context building stage."""
    with pipeline_span(tracer, ctx, "context_build") as span:
        time.sleep(0.003)
        selected = min(input_count, 5)
        tokens_used = selected * 350
        token_budget = 4096
        span.add_event("budget_check", {"tokens_used": tokens_used, "budget": token_budget})
        record_stage_result(
            span,
            outcome="SUCCESS",
            input_count=input_count,
            output_count=selected,
            extra={
                "tokens_used": tokens_used,
                "token_budget": token_budget,
                "excluded_count": input_count - selected,
            },
        )
    return selected


def simulate_generation(
    tracer: Tracer, ctx: ObservabilityContext, context_chunks: int
) -> None:
    """Simulate generation stage."""
    with pipeline_span(tracer, ctx, "generation") as span:
        span.add_event("llm_call_start", {"model": "mock-v1"})
        time.sleep(0.025)
        span.add_event("llm_call_end", {"model": "mock-v1"})
        record_stage_result(
            span,
            outcome="SUCCESS",
            input_count=context_chunks,
            output_count=1,
            extra={
                "model_id": "mock-v1",
                "prompt_tokens": context_chunks * 350 + 200,
                "completion_tokens": 180,
                "citation_count": context_chunks,
            },
        )


def simulate_failed_stage(tracer: Tracer, ctx: ObservabilityContext) -> None:
    """Demonstrate error handling in a span."""
    try:
        with pipeline_span(tracer, ctx, "retrieval", attributes={"mode": "dense"}) as span:
            span.set_attribute("store_id", "qdrant-broken")
            raise ConnectionError("Vector store connection refused")
    except ConnectionError:
        pass  # Handled -- span captured the error


def main() -> None:
    # Reset module-level metric singletons to ensure clean state
    pipeline_latency.reset()
    pipeline_errors.reset()
    pipeline_throughput.reset()

    collector = InMemoryCollector()
    tracer = Tracer(collector=collector, service_name="drifter")
    ctx = tracer.create_context(trace_id="trace-demo-001")

    print("=" * 70)
    print("Drifter Observability -- Multi-Stage Pipeline Trace")
    print("=" * 70)
    print(f"Trace ID: {ctx.trace_id}")
    print()

    # Run the pipeline stages sequentially
    candidate_count = simulate_retrieval(tracer, ctx)
    ranked_count = simulate_reranking(tracer, ctx, candidate_count)
    selected_count = simulate_context_build(tracer, ctx, ranked_count)
    simulate_generation(tracer, ctx, selected_count)

    # -- Span inspection --
    print("--- Collected Spans ---")
    print(f"Total spans: {collector.count}")
    print()

    for i, span in enumerate(collector.spans):
        parent_info = f"parent={span.parent_span_id}" if span.parent_span_id else "root"
        print(f"  [{i + 1}] {span.name}")
        print(f"      span_id={span.span_id}  {parent_info}")
        print(f"      status={span.status.value}  duration={span.duration_ms:.2f}ms")
        if span.attributes:
            filtered = {
                k: v for k, v in span.attributes.items()
                if k not in ("service.name", "pipeline.stage")
            }
            if filtered:
                print(f"      attributes={filtered}")
        if span.events:
            event_names = [e["name"] for e in span.events]
            print(f"      events={event_names}")
        print()

    # -- Parent-child chain --
    print("--- Span Tree (parent-child chain) ---")
    span_map = {s.span_id: s for s in collector.spans}
    for span in collector.spans:
        depth = 0
        current = span
        while current.parent_span_id and current.parent_span_id in span_map:
            depth += 1
            current = span_map[current.parent_span_id]
        indent = "  " * depth
        print(f"  {indent}{span.name} ({span.duration_ms:.2f}ms)")
    print()

    # -- Verify all spans share the same trace_id --
    trace_ids = {s.trace_id for s in collector.spans}
    print(f"All spans share trace_id: {len(trace_ids) == 1} (unique IDs: {trace_ids})")
    print()

    # -- Demonstrate error handling --
    print("--- Error Handling Demo ---")
    error_ctx = tracer.create_context(trace_id="trace-error-001")
    simulate_failed_stage(tracer, error_ctx)

    error_spans = collector.find_by_trace("trace-error-001")
    for span in error_spans:
        print(f"  {span.name}: status={span.status.value}")
        print(f"    error_message={span.error_message}")
        for event in span.events:
            if event["name"] == "exception":
                print(f"    exception.type={event['attributes']['exception.type']}")
                print(f"    exception.message={event['attributes']['exception.message']}")
    print()

    # -- Metrics --
    print("--- Pipeline Metrics ---")
    print(f"pipeline_throughput (successful stages): {pipeline_throughput.value}")
    print(f"pipeline_errors (failed stages):        {pipeline_errors.value}")
    print()

    print("pipeline_latency (stage durations):")
    print(f"  recordings: {pipeline_latency.count}")
    print(f"  total:      {pipeline_latency.sum:.2f}ms")
    if pipeline_latency.count > 0:
        print(f"  p50:        {pipeline_latency.percentile(50):.2f}ms")
        print(f"  p90:        {pipeline_latency.percentile(90):.2f}ms")
        print(f"  p99:        {pipeline_latency.percentile(99):.2f}ms")
    print()

    print("Individual stage latencies:")
    for span in collector.find_by_trace("trace-demo-001"):
        stage = span.attributes.get("pipeline.stage", span.name)
        print(f"  {stage}: {span.duration_ms:.2f}ms")
    print()

    # -- Span export (to_dict) --
    print("--- Span Export (first span as dict) ---")
    first_span = collector.spans[0]
    exported = first_span.to_dict()
    for key in ("name", "trace_id", "span_id", "status", "duration_ms"):
        print(f"  {key}: {exported[key]}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
