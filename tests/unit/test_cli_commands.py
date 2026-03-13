"""Tests for CLI command handlers and main entry point."""

from __future__ import annotations

import json
import os
from io import StringIO
from unittest.mock import patch

import pytest

from apps.cli.errors import EXIT_INPUT_ERROR, EXIT_SUCCESS
from apps.cli.main import _parse_config_overrides, build_parser, main

# Point at nonexistent config/env so tests always use in-memory adapters,
# regardless of whether config.yaml or .env exist in the project root.
_NO_CONFIG = [
    "--config-file", "/dev/null/nonexistent.yaml",
    "--env-file", "/dev/null/nonexistent.env",
]


@pytest.fixture(autouse=True)
def _clean_drifter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all DRIFTER_* env vars so tests use in-memory adapters."""
    for key in list(os.environ):
        if key.startswith("DRIFTER_"):
            monkeypatch.delenv(key, raising=False)


class TestParseConfigOverrides:
    """Test config override parsing."""

    def test_parses_key_value(self) -> None:
        result = _parse_config_overrides(["token_budget=5000"])
        assert result == {"token_budget": "5000"}

    def test_parses_multiple(self) -> None:
        result = _parse_config_overrides(["a=1", "b=2"])
        assert result == {"a": "1", "b": "2"}

    def test_empty(self) -> None:
        result = _parse_config_overrides([])
        assert result == {}

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid config format"):
            _parse_config_overrides(["no_equals"])

    def test_value_with_equals(self) -> None:
        result = _parse_config_overrides(["url=http://host:8080/path?a=b"])
        assert result == {"url": "http://host:8080/path?a=b"}


class TestBuildParser:
    """Test argparse configuration."""

    def test_parser_has_subcommands(self) -> None:
        parser = build_parser()
        # Should not raise when parsing known commands
        args = parser.parse_args(["ask", "test query"])
        assert args.command == "ask"
        assert args.query == "test query"

    def test_parser_global_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--json", "--trace", "t1", "--verbose", "ask", "q"])
        assert args.json is True
        assert args.trace == "t1"
        assert args.verbose is True

    def test_retrieve_command_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["retrieve", "test", "--top-k", "10", "--mode", "dense"])
        assert args.command == "retrieve"
        assert args.query == "test"
        assert args.top_k == 10
        assert args.mode == "dense"

    def test_rerank_command_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["rerank", "test", "--top-n", "5"])
        assert args.command == "rerank"
        assert args.top_n == 5

    def test_build_context_command_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["build-context", "test", "--token-budget", "2000"])
        assert args.command == "build-context"
        assert args.token_budget == 2000

    def test_evaluate_command_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--dataset", "data.json", "--k", "5,10"])
        assert args.command == "evaluate"
        assert args.dataset == "data.json"
        assert args.k == "5,10"


class TestMainEntryPoint:
    """Test the main() function end-to-end."""

    def test_no_command_returns_input_error(self) -> None:
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            code = main([])
        assert code == EXIT_INPUT_ERROR

    def test_ask_with_empty_stores(self) -> None:
        """drifter ask should work with no data (returns no_results)."""
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            code = main([*_NO_CONFIG, "ask", "test question"])
        assert code == EXIT_SUCCESS  # no_results maps to success

    def test_ask_json_output(self) -> None:
        """drifter ask --json should produce valid JSON."""
        with patch("sys.stdout", new_callable=StringIO) as stdout, \
             patch("sys.stderr", new_callable=StringIO):
            main([*_NO_CONFIG, "--json", "ask", "test question"])
            output = json.loads(stdout.getvalue())

        assert "trace_id" in output
        assert "outcome" in output
        assert "data" in output

    def test_retrieve_command(self) -> None:
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            code = main([*_NO_CONFIG, "retrieve", "test query"])
        assert code == EXIT_SUCCESS  # no_results maps to success

    def test_debug_query_outputs_json(self) -> None:
        """debug-query always outputs JSON."""
        with patch("sys.stdout", new_callable=StringIO) as stdout, \
             patch("sys.stderr", new_callable=StringIO):
            main([*_NO_CONFIG, "debug-query", "test question"])
            output = json.loads(stdout.getvalue())

        assert "trace_id" in output
        assert "data" in output

    def test_config_secret_rejection(self) -> None:
        """--config with secret fields should fail."""
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            code = main([*_NO_CONFIG, "--config", "api_key=bad", "ask", "test"])
        from apps.cli.errors import EXIT_CONFIG_ERROR
        assert code == EXIT_CONFIG_ERROR

    def test_config_override_works(self) -> None:
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            code = main([*_NO_CONFIG, "--config", "token_budget=5000", "ask", "test"])
        assert code == EXIT_SUCCESS
