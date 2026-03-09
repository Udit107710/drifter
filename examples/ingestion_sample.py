"""Ingestion subsystem usage example.

Demonstrates the full lifecycle: register → ingest → skip → update → tombstone.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from libs.adapters.memory import MemoryCrawlStateRepository, MemorySourceRepository
from libs.ingestion import (
    IngestionService,
    LocalFilesystemConnector,
    SourceConfig,
    SourceType,
    detect,
)


def main() -> None:
    # Wire up components
    source_repo = MemorySourceRepository()
    crawl_state_repo = MemoryCrawlStateRepository()
    connector = LocalFilesystemConnector()
    service = IngestionService(source_repo, crawl_state_repo, connector, detect)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "sample.txt"
        filepath.write_text("Hello, Drifter!")

        # Register source
        config = SourceConfig(
            source_id="example-src",
            uri=str(filepath),
            source_type=SourceType.FILESYSTEM,
        )
        source_repo.add(config)

        # Run 1: New document → INGEST
        print("=== Run 1: Initial ingestion ===")
        results = service.run("run-001")
        for r in results:
            print(f"  {r.work_item.source_id}: {r.work_item.action.value} → {r.outcome.value}")
            if r.source_ref:
                h = r.source_ref.content_hash[:30]
                print(f"    version={r.source_ref.version}, hash={h}...")

        # Run 2: Unchanged → SKIP
        print("\n=== Run 2: No changes ===")
        results = service.run("run-002")
        for r in results:
            print(f"  {r.work_item.source_id}: {r.work_item.action.value} → {r.outcome.value}")

        # Run 3: Modified file → UPDATE
        filepath.write_text("Updated content for Drifter!")
        print("\n=== Run 3: File modified ===")
        results = service.run("run-003")
        for r in results:
            print(f"  {r.work_item.source_id}: {r.work_item.action.value} → {r.outcome.value}")
            if r.source_ref:
                h = r.source_ref.content_hash[:30]
                print(f"    version={r.source_ref.version}, hash={h}...")

        # Run 4: Deleted file → TOMBSTONE
        filepath.unlink()
        print("\n=== Run 4: File deleted ===")
        results = service.run("run-004")
        for r in results:
            print(f"  {r.work_item.source_id}: {r.work_item.action.value} → {r.outcome.value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
