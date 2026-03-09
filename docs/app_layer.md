# Application Layer

The application layer sits above `orchestrators/` and provides user-facing interfaces.

## Architecture

```
apps/cli/          (argparse, thin handlers, output rendering)
  ↓
orchestrators/     (pipeline composition, trace propagation)
  ↓
libs/*             (business logic, protocols)
  ↓
libs/contracts     (pure domain types)
```

## Orchestrators

### ServiceRegistry (`orchestrators/bootstrap.py`)

The composition root. Creates all services from environment configuration:

1. Loads configs from `DRIFTER_*` env vars (`libs/adapters/env`)
2. Applies `--config` overrides (rejects secret fields)
3. Calls adapter factories (`libs/adapters/factory`)
4. Constructs library services
5. Returns `ServiceRegistry` dataclass

When no env vars are set, all services use in-memory/mock implementations. Production mode activates by setting `DRIFTER_QDRANT_HOST`, `DRIFTER_OPENSEARCH_HOSTS`, etc.

### QueryOrchestrator (`orchestrators/query.py`)

Composes the 4-stage query pipeline:

1. **Retrieval** — `RetrievalBroker.run()` → `BrokerResult`
2. **Reranking** — `RerankerService.run()` → `RerankerResult`
3. **Context building** — `ContextBuilderService.run()` → `BuilderResult`
4. **Generation** — `GenerationService.run()` → `GenerationResult`

Each stage is wrapped in `pipeline_span()` for observability. Supports degraded mode: if reranking fails, falls back to retrieval-order candidates.

Partial pipeline methods: `run_retrieve_only()`, `run_through_rerank()`, `run_through_context()`.

### IngestionOrchestrator (`orchestrators/ingestion.py`)

Composes the ingestion pipeline:

1. **Ingestion** — `IngestionService.run()` → `list[IngestionResult]`
2. **Parsing** — `DocumentParser.parse()` → `CanonicalDocument`
3. **Chunking** — `ChunkingStrategy.chunk()` → `list[Chunk]`
4. **Indexing** — `IndexingService.run()` → `IndexingResult`

## CLI

The CLI is a thin presentation layer (`apps/cli/`). It:

- Parses arguments with argparse
- Creates a `ServiceRegistry` via `create_registry()`
- Dispatches to a command handler
- Renders results via `OutputRenderer`

See [CLI Commands](cli_commands.md) for the full command reference.

## Design Principles

- **Orchestrators import only protocols and contracts** from `libs/`. No concrete adapter imports.
- **CLI handlers are thin** — parse args, call orchestrator, render output. No business logic.
- **Trace ID flows everywhere** — from CLI args → orchestrator → every service → every span.
- **Exit codes are structured** — 0=success, 1=partial, 2=failed, 3=config error, 4=input error.
