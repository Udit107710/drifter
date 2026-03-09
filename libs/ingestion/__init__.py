"""Ingestion subsystem.

Responsibilities:
- Source discovery and registration
- Crawl state tracking
- Document fetching and versioning
- Change detection (new, modified, deleted)
- Tombstone management

Boundary: produces SourceDocumentRef. Does not parse or chunk.
"""

from libs.ingestion.change_detector import detect
from libs.ingestion.connectors.filesystem import LocalFilesystemConnector
from libs.ingestion.models import (
    CrawlState,
    CrawlStatus,
    FetchResult,
    IngestionOutcome,
    IngestionResult,
    SourceConfig,
    SourceType,
    WorkAction,
    WorkItem,
)
from libs.ingestion.protocols import CrawlStateRepository, SourceConnector, SourceRepository
from libs.ingestion.service import IngestionService

__all__ = [
    "CrawlState",
    "CrawlStateRepository",
    "CrawlStatus",
    "FetchResult",
    "IngestionOutcome",
    "IngestionResult",
    "IngestionService",
    "LocalFilesystemConnector",
    "SourceConfig",
    "SourceConnector",
    "SourceRepository",
    "SourceType",
    "WorkAction",
    "WorkItem",
    "detect",
]
