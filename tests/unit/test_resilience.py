"""Tests for libs/resilience.py — shared error classification."""

from __future__ import annotations

from libs.resilience import is_transient_error


class TestIsTransientError:
    """Test transient error detection."""

    def test_timeout_error_is_transient(self) -> None:
        assert is_transient_error(TimeoutError("timed out")) is True

    def test_connection_error_is_transient(self) -> None:
        assert is_transient_error(ConnectionError("refused")) is True

    def test_os_error_is_transient(self) -> None:
        assert is_transient_error(OSError("network unreachable")) is True

    def test_value_error_is_permanent(self) -> None:
        assert is_transient_error(ValueError("bad input")) is False

    def test_runtime_error_is_permanent(self) -> None:
        assert is_transient_error(RuntimeError("crash")) is False

    def test_type_error_is_permanent(self) -> None:
        assert is_transient_error(TypeError("wrong type")) is False

    def test_httpx_like_timeout_detected_by_class_name(self) -> None:
        """Errors with httpx-like class names are detected as transient."""

        class ConnectTimeout(Exception):
            pass

        assert is_transient_error(ConnectTimeout()) is True

    def test_httpx_like_read_timeout_detected(self) -> None:
        class ReadTimeout(Exception):
            pass

        assert is_transient_error(ReadTimeout()) is True

    def test_httpx_like_connect_error_detected(self) -> None:
        class ConnectError(Exception):
            pass

        assert is_transient_error(ConnectError()) is True

    def test_unknown_custom_error_is_permanent(self) -> None:
        class MyCustomError(Exception):
            pass

        assert is_transient_error(MyCustomError()) is False
