"""Change detection logic: compares crawl state with current content to decide action."""

from __future__ import annotations

from libs.ingestion.models import CrawlState, CrawlStatus, WorkAction


def detect(
    crawl_state: CrawlState | None,
    current_hash: str | None,
    current_version: int | None = None,
) -> WorkAction:
    """Determine the work action for a source based on crawl state and current content.

    Decision tree:
    1. current_hash is None → TOMBSTONE (source gone)
    2. No crawl state or status is PENDING → INGEST (new source)
    3. Hash matches → SKIP (unchanged)
    4. Hash differs → UPDATE
    """
    if current_hash is None:
        return WorkAction.TOMBSTONE

    if crawl_state is None or crawl_state.status == CrawlStatus.PENDING:
        return WorkAction.INGEST

    if crawl_state.last_content_hash == current_hash:
        return WorkAction.SKIP

    return WorkAction.UPDATE
