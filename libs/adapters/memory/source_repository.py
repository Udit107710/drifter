"""In-memory source repository for testing."""

from __future__ import annotations

from libs.contracts.common import SourceId
from libs.ingestion.models import SourceConfig


class MemorySourceRepository:
    """Dict-backed source configuration store."""

    def __init__(self) -> None:
        self._store: dict[SourceId, SourceConfig] = {}

    def add(self, config: SourceConfig) -> None:
        if config.source_id in self._store:
            raise ValueError(f"Source already registered: {config.source_id}")
        self._store[config.source_id] = config

    def get(self, source_id: SourceId) -> SourceConfig | None:
        return self._store.get(source_id)

    def list_enabled(self) -> list[SourceConfig]:
        return [c for c in self._store.values() if c.enabled]

    def remove(self, source_id: SourceId) -> None:
        self._store.pop(source_id, None)
