# libs/ingestion/

Source discovery, crawl state tracking, and document fetching. The control layer for source truth.

## Boundary

- **Consumes:** Source configurations (SourceConfig)
- **Produces:** SourceDocumentRef + RawDocument
- **Does not:** Parse, chunk, or index documents

## Key Types

| Type | Purpose |
|------|---------|
| `SourceConfig` | Registered data source (ID, URI, type, schedule, auth) |
| `SourceType` | Enum: FILESYSTEM, HTTP, S3, GIT |
| `CrawlState` | Per-source tracking (last hash, version, status, run ID) |
| `WorkAction` | Decision: INGEST, UPDATE, TOMBSTONE, SKIP |
| `WorkItem` | Decision record for a source in a run |
| `FetchResult` | Raw bytes + MIME type from a connector |
| `IngestionResult` | Outcome per work item (success/skipped/failed) |

## Architecture

```
SourceRepository       CrawlStateRepository
      |                       |
      v                       v
  IngestionService.run(run_id)
      |
      ├── list enabled sources
      ├── for each source:
      |     ├── fetch via SourceConnector
      |     ├── detect changes (content hash comparison)
      |     ├── build SourceDocumentRef + RawDocument
      |     └── update CrawlState
      └── return list[IngestionResult]
```

## Protocols

| Protocol | Method | Purpose |
|----------|--------|---------|
| `SourceConnector` | `fetch(config)`, `list_source_ids(config)` | Fetches raw content from external sources |
| `SourceRepository` | `add`, `get`, `list_enabled`, `remove` | Registry of source configurations |
| `CrawlStateRepository` | `get`, `save`, `get_all` | Per-source crawl state persistence |

## Connectors

- `LocalFilesystemConnector` — Reads files from local filesystem paths

## Change Detection

`change_detector.py` compares content hashes between fetches to decide the work action:
- New source → INGEST
- Changed content hash → UPDATE
- Source removed → TOMBSTONE
- Same content hash → SKIP

## Replay Safety

Each run has a `run_id`. CrawlState tracks the last run, enabling idempotent re-runs. If a run is replayed with the same `run_id`, the change detector sees no new changes and skips.

## Testing

Uses in-memory implementations of all three protocols. No filesystem or network access required.
