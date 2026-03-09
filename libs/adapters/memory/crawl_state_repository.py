"""In-memory crawl state repository for testing."""

from __future__ import annotations

from libs.contracts.common import SourceId
from libs.ingestion.models import CrawlState


class MemoryCrawlStateRepository:
    """Dict-backed crawl state store. Save upserts."""

    def __init__(self) -> None:
        self._store: dict[SourceId, CrawlState] = {}

    def get(self, source_id: SourceId) -> CrawlState | None:
        return self._store.get(source_id)

    def save(self, state: CrawlState) -> None:
        self._store[state.source_id] = state

    def get_all(self) -> list[CrawlState]:
        return list(self._store.values())
