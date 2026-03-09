"""Tests for libs/adapters/langfuse/exporter.py — trace naming & timestamps."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from libs.adapters.config import LangfuseConfig
from libs.adapters.langfuse.exporter import (
    LangfuseSpanExporter,
    _add_wall_times,
    _wall_to_ns,
)
from libs.observability.spans import Span, SpanStatus


def _make_config() -> LangfuseConfig:
    return LangfuseConfig(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:3000",
    )


def _make_span(
    name: str,
    trace_id: str = "trace-1",
    span_id: str = "span-1",
    parent_span_id: str | None = None,
    stage: str | None = None,
    status: SpanStatus = SpanStatus.OK,
) -> Span:
    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
    span = Span(
        name=name,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        status=status,
        start_time=100.0,
        start_wall=now,
        end_time=100.05,
        end_wall=datetime(2025, 6, 15, 12, 0, 0, 50000, tzinfo=UTC),
    )
    if stage:
        span.attributes["pipeline.stage"] = stage
    return span


class TestWallToNs:
    def test_converts_datetime_to_nanoseconds(self) -> None:
        dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        ns = _wall_to_ns(dt)
        assert ns is not None
        # 2025-06-15T12:00:00Z epoch seconds * 1e9
        expected = int(dt.timestamp() * 1_000_000_000)
        assert ns == expected

    def test_none_returns_none(self) -> None:
        assert _wall_to_ns(None) is None


class TestAddWallTimes:
    def test_adds_start_and_end_wall(self) -> None:
        span = _make_span("test")
        metadata: dict = {}
        _add_wall_times(metadata, span)
        assert "start_wall" in metadata
        assert "end_wall" in metadata
        assert "duration_ms" in metadata
        assert metadata["duration_ms"] == round(span.duration_ms, 2)

    def test_no_end_wall(self) -> None:
        span = _make_span("test")
        span.end_wall = None
        metadata: dict = {}
        _add_wall_times(metadata, span)
        assert metadata["start_wall"] == span.start_wall.isoformat()
        assert metadata.get("end_wall") is None


class TestTraceNaming:
    """Verify that the trace name is set via propagate_attributes."""

    def test_propagate_attributes_called_with_trace_name(self) -> None:
        exporter = LangfuseSpanExporter(_make_config())
        mock_client = MagicMock()

        # Mock start_as_current_span to return a context manager
        mock_lf_span = MagicMock()
        mock_lf_span.__enter__ = MagicMock(return_value=mock_lf_span)
        mock_lf_span.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_span.return_value = mock_lf_span

        # Mock start_span for child
        mock_child_span = MagicMock()
        mock_client.start_span.return_value = mock_child_span

        exporter._client = mock_client

        # Create root and child
        root = _make_span("rag-pipeline", trace_id="t1", span_id="root")
        child = _make_span(
            "retrieval", trace_id="t1", span_id="child-1",
            parent_span_id="root", stage="retrieval",
        )

        # Buffer the child first (as happens in real flow)
        exporter.collect(child)

        # Patch propagate_attributes and then send root
        with patch(
            "langfuse.propagate_attributes",
        ) as mock_propagate:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_propagate.return_value = mock_ctx

            exporter.collect(root)

            # propagate_attributes must be called with trace_name
            mock_propagate.assert_called_once_with(
                trace_name="rag-pipeline",
            )

        # Root should have been created with correct name
        mock_client.start_as_current_span.assert_called_once()
        create_kwargs = mock_client.start_as_current_span.call_args
        assert create_kwargs.kwargs["name"] == "rag-pipeline"

    def test_no_children_still_propagates_trace_name(self) -> None:
        exporter = LangfuseSpanExporter(_make_config())
        mock_client = MagicMock()

        mock_lf_span = MagicMock()
        mock_lf_span.__enter__ = MagicMock(return_value=mock_lf_span)
        mock_lf_span.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_span.return_value = mock_lf_span

        exporter._client = mock_client

        root = _make_span("rag-pipeline", trace_id="t2", span_id="root")

        with patch(
            "langfuse.propagate_attributes",
        ) as mock_propagate:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_propagate.return_value = mock_ctx

            exporter.collect(root)

            mock_propagate.assert_called_once_with(
                trace_name="rag-pipeline",
            )


class TestChildTimestamps:
    """Verify that child spans pass end_time to Langfuse end()."""

    def test_span_end_called_with_nanoseconds(self) -> None:
        exporter = LangfuseSpanExporter(_make_config())
        mock_client = MagicMock()

        mock_lf_span = MagicMock()
        mock_lf_span.__enter__ = MagicMock(return_value=mock_lf_span)
        mock_lf_span.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_span.return_value = mock_lf_span

        mock_child_lf = MagicMock()
        mock_client.start_span.return_value = mock_child_lf

        exporter._client = mock_client

        root = _make_span("rag-pipeline", trace_id="t1", span_id="root")
        child = _make_span(
            "retrieval", trace_id="t1", span_id="child-1",
            parent_span_id="root", stage="retrieval",
        )

        exporter.collect(child)
        exporter.collect(root)

        # Child span's end() should be called with end_time in nanoseconds
        expected_ns = _wall_to_ns(child.end_wall)
        mock_child_lf.end.assert_called_once_with(end_time=expected_ns)

    def test_generation_end_called_with_nanoseconds(self) -> None:
        exporter = LangfuseSpanExporter(_make_config())
        mock_client = MagicMock()

        mock_lf_span = MagicMock()
        mock_lf_span.__enter__ = MagicMock(return_value=mock_lf_span)
        mock_lf_span.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_span.return_value = mock_lf_span

        mock_gen_lf = MagicMock()
        mock_client.start_observation.return_value = mock_gen_lf

        exporter._client = mock_client

        root = _make_span("rag-pipeline", trace_id="t1", span_id="root")
        gen = _make_span(
            "generation", trace_id="t1", span_id="gen-1",
            parent_span_id="root", stage="generation",
        )

        exporter.collect(gen)
        exporter.collect(root)

        expected_ns = _wall_to_ns(gen.end_wall)
        mock_gen_lf.end.assert_called_once_with(end_time=expected_ns)


class TestMetadataContainsWallTimes:
    """Verify that wall-clock times are included in metadata."""

    def test_child_metadata_has_wall_times(self) -> None:
        exporter = LangfuseSpanExporter(_make_config())
        mock_client = MagicMock()

        mock_lf_span = MagicMock()
        mock_lf_span.__enter__ = MagicMock(return_value=mock_lf_span)
        mock_lf_span.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_span.return_value = mock_lf_span

        mock_child_lf = MagicMock()
        mock_client.start_span.return_value = mock_child_lf

        exporter._client = mock_client

        root = _make_span("rag-pipeline", trace_id="t1", span_id="root")
        child = _make_span(
            "retrieval", trace_id="t1", span_id="child-1",
            parent_span_id="root", stage="retrieval",
        )

        exporter.collect(child)
        exporter.collect(root)

        # Check metadata passed to start_span
        child_call = mock_client.start_span.call_args
        metadata = child_call.kwargs["metadata"]
        assert "start_wall" in metadata
        assert "end_wall" in metadata
        assert "duration_ms" in metadata

    def test_root_metadata_has_wall_times(self) -> None:
        exporter = LangfuseSpanExporter(_make_config())
        mock_client = MagicMock()

        mock_lf_span = MagicMock()
        mock_lf_span.__enter__ = MagicMock(return_value=mock_lf_span)
        mock_lf_span.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_span.return_value = mock_lf_span

        exporter._client = mock_client

        root = _make_span("rag-pipeline", trace_id="t1", span_id="root")
        exporter.collect(root)

        root_call = mock_client.start_as_current_span.call_args
        metadata = root_call.kwargs["metadata"]
        assert "start_wall" in metadata
        assert "end_wall" in metadata
        assert "duration_ms" in metadata
