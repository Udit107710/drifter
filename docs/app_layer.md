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
4. Constructs library services (query pipeline, ingestion pipeline)
5. Returns `ServiceRegistry` dataclass

The registry wires both the **query pipeline** (retrieval → reranking → context → generation) and the **ingestion pipeline** (ingest → parse → chunk → embed → index). Store-backed index writers (`libs/adapters/store_writers.py`) bridge the `VectorStore`/`LexicalStore` protocols to the `VectorIndexWriter`/`LexicalIndexWriter` protocols, so the `IndexingService` writes directly to real stores.

When no env vars are set, all services use in-memory/mock implementations. Production mode activates by setting `DRIFTER_QDRANT_HOST`, `DRIFTER_OPENSEARCH_HOSTS`, etc. The CLI auto-loads `.env` from the project root via `python-dotenv`.

**LLM provider selection**: With `config.yaml`, set `generation.provider` explicitly (`ollama`, `vllm`, `openai`, `openrouter`, `gemini`). In env-var fallback mode, priority is: OpenRouter > OpenAI > Gemini > Ollama. If none is configured, `MockGenerator` is used.

**Reranker priority**: When `DRIFTER_TEI_RERANKER_URL` is set and the TEI server is reachable, `TeiCrossEncoderReranker` is used for neural reranking. Otherwise, if `DRIFTER_HF_TOKEN` is set and the HuggingFace Inference API is reachable, `HuggingFaceReranker` is used. Final fallback is `FeatureBasedReranker` (local, no external calls).

**Token counting**: The bootstrap automatically uses `TiktokenTokenCounter` (accurate BPE token counts) when tiktoken is installed, falling back to `WhitespaceTokenCounter` otherwise. Install tiktoken with `uv sync --extra tokenizers`.

**Default tuning**: Token budget is 5000, lexical RRF weight is 1.5, and `max_chunks_per_source=2` in the context builder to ensure source diversity.

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

1. **Ingestion** — `IngestionService.run()` → `list[IngestionResult]` (fetch files, detect changes)
2. **Parsing** — `DocumentParser.parse()` → `CanonicalDocument` (Markdown or plain text)
3. **Chunking** — `ChunkingStrategy.chunk()` → `list[Chunk]` (recursive structure-aware)
4. **Indexing** — `IndexingService.run()` → `IndexingResult` (embed + write to vector and lexical stores)

The CLI `rag ingest --path <dir>` command registers each file as a source, then runs the orchestrator. Sources are tracked with content hashing for change detection and replay safety.

## CLI

The CLI is a thin presentation layer (`apps/cli/`). It:

- Parses arguments with argparse
- Creates a `ServiceRegistry` via `create_registry()`
- Dispatches to a command handler
- Renders results via `OutputRenderer`

See [CLI Commands](cli_commands.md) for the full command reference.

## Design Principles

- **Adapter lifecycle via protocols** — the bootstrap uses `isinstance(obj, Connectable)` and `isinstance(obj, HealthCheckable)` (from `libs/adapters/protocols.py`) instead of `hasattr` checks, maintaining type safety throughout the composition root.
- **Orchestrators import only protocols and contracts** from `libs/`. No concrete adapter imports.
- **CLI handlers are thin** — parse args, call orchestrator, render output. No business logic.
- **Trace ID flows everywhere** — from CLI args → orchestrator → every service → every span.
- **Exit codes are structured** — 0=success, 1=partial, 2=failed, 3=config error, 4=input error.
