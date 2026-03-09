"""Tests for apps/cli/output.py — OutputRenderer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from io import StringIO
from unittest.mock import patch

from apps.cli.output import OutputRenderer, _serialize
from orchestrators.query import QueryResult


class TestSerialize:
    """Test the _serialize helper."""

    def test_primitives(self) -> None:
        assert _serialize("hello") == "hello"
        assert _serialize(42) == 42
        assert _serialize(3.14) == 3.14
        assert _serialize(True) is True
        assert _serialize(None) is None

    def test_enum(self) -> None:
        class Color(Enum):
            RED = "red"
        assert _serialize(Color.RED) == "red"

    def test_datetime(self) -> None:
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = _serialize(dt)
        assert "2025-01-01" in result

    def test_dict(self) -> None:
        result = _serialize({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_list(self) -> None:
        result = _serialize([1, "two", 3])
        assert result == [1, "two", 3]

    def test_bytes(self) -> None:
        result = _serialize(b"hello")
        assert "5 bytes" in result

    def test_nested(self) -> None:
        class Color(Enum):
            RED = "red"
        result = _serialize({"colors": [Color.RED]})
        assert result == {"colors": ["red"]}


class TestOutputRendererJsonMode:
    """Test JSON output mode."""

    def test_render_query_result_json(self) -> None:
        renderer = OutputRenderer(json_mode=True)
        result = QueryResult(
            trace_id="trace-1",
            query="test query",
            outcome="no_results",
            total_latency_ms=42.5,
        )

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            renderer.render_query_result(result)
            output = json.loads(stdout.getvalue())

        assert output["trace_id"] == "trace-1"
        assert output["outcome"] == "no_results"
        assert output["latency_ms"] == 42.5
        assert output["query"] == "test query"
        assert "data" in output
        assert "errors" in output

    def test_render_error_json(self) -> None:
        renderer = OutputRenderer(json_mode=True)

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            renderer.render_error("something broke", trace_id="t1")
            output = json.loads(stdout.getvalue())

        assert output["outcome"] == "error"
        assert output["errors"] == ["something broke"]
        assert output["trace_id"] == "t1"


class TestOutputRendererHumanMode:
    """Test human-readable output mode."""

    def test_render_query_result_human(self) -> None:
        renderer = OutputRenderer(json_mode=False)
        result = QueryResult(
            trace_id="trace-1",
            query="test query",
            outcome="no_results",
            total_latency_ms=42.5,
        )

        with patch("sys.stderr", new_callable=StringIO) as stderr, \
             patch("sys.stdout", new_callable=StringIO) as stdout:
            renderer.render_query_result(result)
            err = stderr.getvalue()
            out = stdout.getvalue()

        assert "trace-1" in err
        assert "no_results" in err
        assert "No results found" in out

    def test_render_error_human(self) -> None:
        renderer = OutputRenderer(json_mode=False)

        with patch("sys.stderr", new_callable=StringIO) as stderr:
            renderer.render_error("something broke", trace_id="t1")
            err = stderr.getvalue()

        assert "something broke" in err
        assert "t1" in err


class TestExitCodes:
    """Test exit code mapping."""

    def test_exit_codes(self) -> None:
        from apps.cli.errors import (
            EXIT_FAILED,
            EXIT_PARTIAL,
            EXIT_SUCCESS,
            outcome_to_exit_code,
        )

        assert outcome_to_exit_code("success") == EXIT_SUCCESS
        assert outcome_to_exit_code("partial") == EXIT_PARTIAL
        assert outcome_to_exit_code("no_results") == EXIT_SUCCESS
        assert outcome_to_exit_code("failed") == EXIT_FAILED
        assert outcome_to_exit_code("unknown") == EXIT_FAILED
