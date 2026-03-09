"""Shared resilience utilities: error classification and transient detection."""

from __future__ import annotations


def is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* is likely transient and worth retrying.

    Covers stdlib network errors and common httpx timeout/connection errors.
    """
    # Stdlib transient errors
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    # httpx transient errors (check by class name to avoid hard dependency)
    return type(exc).__name__ in (
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "RemoteProtocolError",
    )
