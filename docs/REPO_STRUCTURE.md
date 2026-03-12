# Repository Structure

## Top-level directories

| Directory | Purpose |
|---|---|
| `apps/` | Application entry points that expose orchestrators to the outside world |
| `apps/cli/` | Command-line interface (`rag` command). Thin argparse handlers, output rendering |
| `apps/api/` | Query-serving HTTP API (planned). Online, latency-sensitive, read-heavy |
| `apps/worker/` | Background ingestion worker (planned). Offline, batch-oriented, write-heavy |
| `apps/debugger_ui/` | Pipeline inspection UI (planned) for development and debugging |
| `orchestrators/` | Thin composition layers that wire libraries into pipelines |
| `libs/` | Pure libraries — one per subsystem, strictly independent |
| `libs/contracts/` | Shared domain models. Every other package depends on this; it depends on nothing |
| `libs/ingestion/` | Source discovery, crawl state, document fetching, versioning, tombstones |
| `libs/parsing/` | Raw bytes → CanonicalDocument with structured Blocks |
| `libs/chunking/` | CanonicalDocument → list[Chunk] with lineage and metadata |
| `libs/embeddings/` | Chunk → ChunkEmbedding via provider-agnostic protocol |
| `libs/indexing/` | Write chunks and embeddings to vector/lexical indexes |
| `libs/retrieval/stores/` | VectorStore and LexicalStore protocols |
| `libs/retrieval/broker/` | Orchestrates dense, lexical, and hybrid retrieval |
| `libs/reranking/` | Score and reorder candidates (cross-encoder, feature-based) |
| `libs/context_builder/` | Token budgeting, evidence selection, redundancy removal |
| `libs/generation/` | Prompt construction, LLM call, citation extraction |
| `libs/evaluation/` | Retrieval and answer quality metrics, dataset management |
| `libs/observability/` | OpenTelemetry span helpers and structured metrics |
| `libs/adapters/` | Concrete implementations for external services (Qdrant, OpenSearch, TEI, Ollama, vLLM, OpenAI, OpenRouter, Gemini, HuggingFace, Langfuse, OTel) |
| `libs/adapters/protocols.py` | Adapter lifecycle protocols (`Connectable`, `HealthCheckable`) |
| `libs/adapters/memory/` | In-memory implementations for deterministic local testing |
| `libs/experiments/` | Reproducible experiment configuration, execution, and comparison |
| `experiments/` | Experiment workspace, organized by subsystem |
| `tests/` | Unit and integration tests |
| `tests/fixtures/` | Fixture documents and gold-standard evaluation sets |
| `scripts/` | Development and operational scripts |
| `docs/` | Architecture docs and design documents |
| `prompts/` | Numbered implementation prompts (build sequence) |

## Dependency direction

```
apps/           (HTTP/CLI — knows orchestrators and factory wiring)
  ↓
orchestrators/  (knows library protocols + contracts)
  ↓
libs/*          (knows only libs/contracts)
  ↓
libs/contracts  (knows nothing — pure domain types)
```

No upward imports. No circular dependencies.

## Key files

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, tool configuration |
| `Makefile` | Common development commands (install, test, lint, fmt) |
| `docker-compose.yml` | Optional infrastructure for real backends (Qdrant, OpenSearch, Langfuse, Jaeger, Redis) |
| `.env.example` | Configuration template — copy to `.env` for local use |
| `AGENTS.md` | Operational rules for AI agents working in this repo |
| `MASTER_PROMPT.md` | Persona and working rules for Claude Code |
| `CLAUDE.md` | Quick-reference guidance for Claude Code |
| `orchestrators/bootstrap.py` | Composition root — creates ServiceRegistry from env vars |
| `orchestrators/query.py` | QueryOrchestrator — 4-stage sync query pipeline |
| `orchestrators/async_query.py` | AsyncQueryOrchestrator — async retrieval, sync stages 2-4 |
| `orchestrators/ingestion.py` | IngestionOrchestrator — ingest/parse/chunk/index pipeline |
| `libs/adapters/factory.py` | Adapter factory functions with in-memory fallbacks (all return typed Protocols) |
| `libs/resilience.py` | Transient error classification, `RetryConfig`, `resilient_call()` / `async_resilient_call()` with exponential backoff and jitter |
| `libs/adapters/env.py` | Environment variable loaders for all providers |
| `libs/adapters/config.py` | Configuration dataclasses for all adapters |
| `examples/` | 14 runnable sample scripts demonstrating each subsystem |
