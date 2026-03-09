"""Ingestion subsystem models: source config, crawl state, work items, fetch results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from libs.contracts.common import RunId, SourceId
from libs.contracts.documents import RawDocument, SourceDocumentRef


class SourceType(Enum):
    """Supported source connector types."""

    FILESYSTEM = "filesystem"
    HTTP = "http"
    S3 = "s3"
    GIT = "git"


@dataclass(frozen=True)
class SourceConfig:
    """A registered data source definition."""

    source_id: SourceId
    uri: str
    source_type: SourceType
    schedule: str | None = None
    auth: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    registered_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.uri:
            raise ValueError("uri must not be empty")


class CrawlStatus(Enum):
    """Status of the last crawl for a source."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"


@dataclass
class CrawlState:
    """Per-source crawl tracking state. Mutable — represents a DB row updated in-place."""

    source_id: SourceId
    last_content_hash: str | None = None
    last_version: int = 0
    last_crawl_at: datetime | None = None
    status: CrawlStatus = CrawlStatus.PENDING
    run_id: RunId | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")


class WorkAction(Enum):
    """Decision for what to do with a source in a crawl run."""

    INGEST = "ingest"
    UPDATE = "update"
    TOMBSTONE = "tombstone"
    SKIP = "skip"


@dataclass(frozen=True)
class WorkItem:
    """Decision record for a single source in a run."""

    source_id: SourceId
    action: WorkAction
    run_id: RunId
    current_hash: str | None = None
    current_version: int | None = None
    previous_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.run_id:
            raise ValueError("run_id must not be empty")


@dataclass(frozen=True)
class FetchResult:
    """Raw output from a connector before the service builds SourceDocumentRef.

    Connectors only do I/O — they don't compute hashes or version numbers.
    """

    raw_bytes: bytes
    mime_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.raw_bytes:
            raise ValueError("raw_bytes must not be empty")
        if not self.mime_type:
            raise ValueError("mime_type must not be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")


class IngestionOutcome(Enum):
    """Outcome of processing a single work item."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestionResult:
    """Result of processing one work item during an ingestion run."""

    work_item: WorkItem
    outcome: IngestionOutcome
    raw_document: RawDocument | None = None
    source_ref: SourceDocumentRef | None = None
    error: str | None = None
    completed_at: datetime | None = None
