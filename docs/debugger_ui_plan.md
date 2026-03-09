# Debugger UI Plan

The debugger UI will provide a browser-based interface for inspecting the RAG pipeline. It reuses the same orchestrators as the CLI.

## Architecture

```
apps/debugger_ui/     (web framework, templates)
  ↓
orchestrators/        (same QueryOrchestrator, IngestionOrchestrator)
  ↓
libs/*                (business logic)
```

## Planned Features

1. **Query Explorer** — Run queries and see all intermediate results (candidates, rerank scores, context pack, citations) side by side.

2. **Chunk Inspector** — Browse indexed chunks with their lineage, metadata, and embedding vectors.

3. **Trace Viewer** — Visualize span trees for any trace ID. Show per-stage latency waterfall.

4. **Evaluation Dashboard** — Compare experiment runs. Show metric deltas with visual indicators.

5. **Index Browser** — View store contents, document counts, embedding stats.

## Implementation Notes

- The debugger UI is a development/debugging tool, not a production API.
- It will use `QueryOrchestrator` directly — no separate API layer needed.
- All data comes from the same `ServiceRegistry` as the CLI.
- The `debug-query` CLI command already produces the full data structure the UI needs.
- Framework choice deferred (likely a lightweight ASGI app).

## Prerequisites

- All orchestrators implemented (done)
- CLI working with all commands (done)
- HTTP API layer (future: `apps/api/`)
