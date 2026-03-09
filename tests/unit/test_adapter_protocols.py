"""Tests for adapter lifecycle protocols (Connectable, HealthCheckable)."""

from __future__ import annotations

from libs.adapters.protocols import Connectable, HealthCheckable


class _FakeConnectable:
    def connect(self) -> None:
        pass


class _FakeHealthCheckable:
    def health_check(self) -> bool:
        return True


class _PlainObject:
    pass


class TestConnectable:
    def test_detects_connectable(self) -> None:
        assert isinstance(_FakeConnectable(), Connectable)

    def test_rejects_non_connectable(self) -> None:
        assert not isinstance(_PlainObject(), Connectable)


class TestHealthCheckable:
    def test_detects_health_checkable(self) -> None:
        assert isinstance(_FakeHealthCheckable(), HealthCheckable)

    def test_rejects_non_health_checkable(self) -> None:
        assert not isinstance(_PlainObject(), HealthCheckable)
