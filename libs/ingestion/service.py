"""Ingestion service: orchestrates source fetching, change detection, and work item production."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime

from libs.contracts.common import RunId
from libs.contracts.documents import RawDocument, SourceDocumentRef
from libs.ingestion.models import (
    CrawlState,
    CrawlStatus,
    IngestionOutcome,
    IngestionResult,
    SourceConfig,
    WorkAction,
    WorkItem,
)
from libs.ingestion.protocols import CrawlStateRepository, SourceConnector, SourceRepository


class IngestionService:
    """Orchestrates the ingestion pipeline: fetch → detect changes → produce documents."""

    def __init__(
        self,
        source_repo: SourceRepository,
        crawl_state_repo: CrawlStateRepository,
        connector: SourceConnector,
        change_detector: Callable[..., WorkAction],
    ) -> None:
        self._source_repo = source_repo
        self._crawl_state_repo = crawl_state_repo
        self._connector = connector
        self._detect = change_detector

    def run(self, run_id: RunId) -> list[IngestionResult]:
        """Execute an ingestion run across all enabled sources."""
        sources = self._source_repo.list_enabled()
        results: list[IngestionResult] = []

        for config in sources:
            try:
                result = self._process_source(config, run_id)
            except Exception as exc:
                # Record failure but continue with other sources
                work_item = WorkItem(
                    source_id=config.source_id,
                    action=WorkAction.SKIP,
                    run_id=run_id,
                )
                crawl_state = self._crawl_state_repo.get(config.source_id)
                if crawl_state is None:
                    crawl_state = CrawlState(source_id=config.source_id)
                crawl_state.status = CrawlStatus.FAILED
                crawl_state.run_id = run_id
                crawl_state.error_message = str(exc)
                crawl_state.last_crawl_at = datetime.now(UTC)
                self._crawl_state_repo.save(crawl_state)
                result = IngestionResult(
                    work_item=work_item,
                    outcome=IngestionOutcome.FAILED,
                    error=str(exc),
                    completed_at=datetime.now(UTC),
                )
            results.append(result)

        return results

    def _process_source(self, config: SourceConfig, run_id: RunId) -> IngestionResult:
        """Process a single source: fetch, detect, build documents."""
        crawl_state = self._crawl_state_repo.get(config.source_id)

        # Replay safety: already completed in this run
        if (
            crawl_state is not None
            and crawl_state.run_id == run_id
            and crawl_state.status == CrawlStatus.COMPLETED
        ):
            work_item = WorkItem(
                source_id=config.source_id,
                action=WorkAction.SKIP,
                run_id=run_id,
                current_hash=crawl_state.last_content_hash,
            )
            return IngestionResult(
                work_item=work_item,
                outcome=IngestionOutcome.SKIPPED,
                completed_at=datetime.now(UTC),
            )

        # Fetch from connector
        fetch_result = self._connector.fetch(config)

        # Compute hash (None if source gone)
        content_hash = (
            self._compute_content_hash(fetch_result.raw_bytes)
            if fetch_result is not None
            else None
        )

        # Determine current version
        current_version = (
            (crawl_state.last_version + 1) if crawl_state and crawl_state.last_version else 1
        )

        # Detect action
        action = self._detect(crawl_state, content_hash, current_version)

        # Build work item
        work_item = WorkItem(
            source_id=config.source_id,
            action=action,
            run_id=run_id,
            current_hash=content_hash,
            current_version=current_version,
            previous_hash=crawl_state.last_content_hash if crawl_state else None,
        )

        now = datetime.now(UTC)
        source_ref: SourceDocumentRef | None = None
        raw_document: RawDocument | None = None

        if action == WorkAction.SKIP:
            outcome = IngestionOutcome.SKIPPED
        elif action in (WorkAction.INGEST, WorkAction.UPDATE):
            assert fetch_result is not None
            assert content_hash is not None

            version = 1 if action == WorkAction.INGEST else (
                crawl_state.last_version + 1 if crawl_state else 1
            )

            source_ref = SourceDocumentRef(
                source_id=config.source_id,
                uri=config.uri,
                content_hash=content_hash,
                fetched_at=now,
                version=version,
            )
            raw_document = RawDocument(
                source_ref=source_ref,
                raw_bytes=fetch_result.raw_bytes,
                mime_type=fetch_result.mime_type,
                size_bytes=fetch_result.size_bytes,
            )
            outcome = IngestionOutcome.SUCCESS
        elif action == WorkAction.TOMBSTONE:
            outcome = IngestionOutcome.SUCCESS
        else:
            outcome = IngestionOutcome.SKIPPED

        # Update crawl state
        if crawl_state is None:
            crawl_state = CrawlState(source_id=config.source_id)

        if action == WorkAction.TOMBSTONE:
            crawl_state.status = CrawlStatus.TOMBSTONED
        else:
            crawl_state.status = CrawlStatus.COMPLETED

        if content_hash is not None:
            crawl_state.last_content_hash = content_hash
        if action in (WorkAction.INGEST, WorkAction.UPDATE):
            assert source_ref is not None
            crawl_state.last_version = source_ref.version
        crawl_state.last_crawl_at = now
        crawl_state.run_id = run_id
        crawl_state.error_message = None
        self._crawl_state_repo.save(crawl_state)

        return IngestionResult(
            work_item=work_item,
            outcome=outcome,
            raw_document=raw_document,
            source_ref=source_ref,
            completed_at=now,
        )

    @staticmethod
    def _compute_content_hash(raw_bytes: bytes) -> str:
        """Compute a sha256 content hash."""
        return f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"
