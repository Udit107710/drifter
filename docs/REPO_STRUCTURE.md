# Repository Structure

## Top-level directories

| Directory | Purpose |
|---|---|
| `apps/` | Application entry points that expose orchestrators to the outside world |
| `apps/api/` | Query-serving HTTP API. Online, latency-sensitive, read-heavy |
| `apps/worker/` | Background ingestion worker. Offline, batch-oriented, write-heavy |
| `apps/debugger_ui/` | Pipeline inspection UI for development and debugging |
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
| `libs/adapters/` | Concrete implementations for external services (Qdrant, OpenSearch, TEI, vLLM) |
| `libs/adapters/memory/` | In-memory implementations for deterministic local testing |
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
| `docker-compose.yml` | Optional infrastructure for real backends (Qdrant, OpenSearch, Jaeger) |
| `.env.example` | Configuration template — copy to `.env` for local use |
| `AGENTS.md` | Operational rules for AI agents working in this repo |
| `MASTER_PROMPT.md` | Persona and working rules for Claude Code |
| `CLAUDE.md` | Quick-reference guidance for Claude Code |
