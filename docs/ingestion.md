# Ingestion Subsystem

## Overview

The ingestion subsystem is the control layer for source truth. It registers data sources, tracks crawl state, detects changes, and produces `SourceDocumentRef` + `RawDocument` for downstream parsing. It does **not** parse, chunk, or index.

## Lifecycle

```
Source Registration → Crawl → Fetch → Change Detection → Work Items → Produce Documents
```

1. **Register** a `SourceConfig` in the `SourceRepository` (source_id, URI, type, schedule)
2. **Run** the `IngestionService` with a `RunId`
3. For each enabled source:
   - **Fetch** via `SourceConnector` → `FetchResult` (raw bytes + MIME type)
   - **Hash** content centrally (`sha256:<hex>`)
   - **Detect** changes by comparing hash against `CrawlState` → `WorkAction`
   - **Build** `SourceDocumentRef` and `RawDocument` if needed
   - **Update** `CrawlState` with new hash, version, status, run_id

## Change Detection

The `detect()` function is pure logic with no I/O:

| Condition | Action |
|-----------|--------|
| Source gone (no fetch result) | `TOMBSTONE` |
| No crawl state or status=PENDING | `INGEST` |
| Hash unchanged | `SKIP` |
| Hash differs | `UPDATE` |

Content hash is the primary signal. Version numbers are recorded but not used for detection (sources may not provide them).

## Failure Handling

- If a connector raises during fetch, the crawl state is set to `FAILED` with the error message
- Failed sources are retried on the next run (status reverts to eligible for processing)
- Other sources in the same run continue processing — one failure doesn't block the batch

## Idempotency

- **Same content** = `SKIP` — no redundant processing
- **Replay safety** — if a source's crawl state already shows `COMPLETED` for the current `run_id`, it is skipped. This makes runs safe to retry after partial failures.

## Tombstone Lifecycle

1. Source is deleted (connector returns `None`)
2. Crawl state is set to `TOMBSTONED`
3. Downstream cleanup (removing chunks, embeddings, index entries) is the responsibility of the orchestrator — not the ingestion subsystem

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│SourceRepo    │     │IngestionService │     │SourceConnector │
│(registry)    │────▶│(orchestrator)   │────▶│(I/O only)      │
└──────────────┘     └────────┬────────┘     └────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ ChangeDetector    │
                    │ (pure logic)      │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │CrawlStateRepo     │
                    │(state tracking)   │
                    └───────────────────┘
```

### Key design decisions

- **Connectors return `FetchResult`, not `RawDocument`** — building `SourceDocumentRef` requires version numbers and content hashes, which depend on crawl state. Connectors only do I/O.
- **`CrawlState` is mutable** — it represents a DB row updated in-place during a run. Frozen would require creating new instances for every field change on what is fundamentally mutable state.
- **Hashing is centralized** in the service, not in connectors — ensures consistency regardless of connector implementation.

## Usage

```python
from libs.adapters.memory import MemoryCrawlStateRepository, MemorySourceRepository
from libs.ingestion import (
    IngestionService,
    LocalFilesystemConnector,
    SourceConfig,
    SourceType,
    detect,
)

# Wire up
source_repo = MemorySourceRepository()
crawl_state_repo = MemoryCrawlStateRepository()
connector = LocalFilesystemConnector()
service = IngestionService(source_repo, crawl_state_repo, connector, detect)

# Register a source
config = SourceConfig(
    source_id="my-source",
    uri="/path/to/document.pdf",
    source_type=SourceType.FILESYSTEM,
)
source_repo.add(config)

# Run ingestion
results = service.run("run-001")
for result in results:
    print(f"{result.work_item.action.value}: {result.outcome.value}")
    if result.raw_document:
        # Pass to parsing subsystem
        pass
```

## Extending

To add a new source type:

1. Implement `SourceConnector` protocol (two methods: `fetch` and `list_source_ids`)
2. Add a `SourceType` enum value
3. Inject the connector into `IngestionService`
