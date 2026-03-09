"""Ingestion subsystem protocols: repository and connector interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.common import SourceId
from libs.ingestion.models import CrawlState, FetchResult, SourceConfig


@runtime_checkable
class SourceRepository(Protocol):
    """Registry of data source configurations."""

    def add(self, config: SourceConfig) -> None: ...
    def get(self, source_id: SourceId) -> SourceConfig | None: ...
    def list_enabled(self) -> list[SourceConfig]: ...
    def remove(self, source_id: SourceId) -> None: ...


@runtime_checkable
class CrawlStateRepository(Protocol):
    """Persistence for per-source crawl state."""

    def get(self, source_id: SourceId) -> CrawlState | None: ...
    def save(self, state: CrawlState) -> None: ...
    def get_all(self) -> list[CrawlState]: ...


@runtime_checkable
class SourceConnector(Protocol):
    """Fetches raw content from external sources."""

    def fetch(self, config: SourceConfig) -> FetchResult | None:
        """Fetch content from the source. Returns None if the source is gone."""
        ...

    def list_source_ids(self, config: SourceConfig) -> list[str]:
        """List available source identifiers (for bulk deletion detection)."""
        ...
