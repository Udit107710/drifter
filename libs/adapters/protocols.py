"""Adapter lifecycle protocols: connection and health checking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Connectable(Protocol):
    """Adapter that requires an explicit connection step."""

    def connect(self) -> None: ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Adapter that supports health checking."""

    def health_check(self) -> bool: ...
