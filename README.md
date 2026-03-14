# Drifter

Modular, production-grade Retrieval Augmented Generation (RAG) system built from first principles.

Drifter treats RAG as a four-stage information retrieval pipeline with strict subsystem boundaries, protocol-based abstractions, and full observability. It is designed for learning, experimentation, and scalable deployment.

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install in development mode
make dev

# Run tests (all local, no external services needed)
make test

# Lint and type check
make lint
make typecheck
```

### CLI

Drifter ships a `drifter` CLI that works with zero external services:

```bash
# Ask a question (uses in-memory/mock services by default)
uv run drifter ask "What is machine learning?"

# Stream tokens in real-time (Ollama NDJSON, vLLM SSE)
uv run drifter ask --stream "What is machine learning?"

# JSON output
uv run drifter --json ask "What is machine learning?"

# Run individual pipeline stages
uv run drifter retrieve "query" --mode dense --top-k 10
uv run drifter rerank "query" --top-k 10
uv run drifter build-context "query" --token-budget 2000

# Full debug output (always JSON)
uv run drifter debug-query "query"

# Run evaluation
uv run drifter evaluate --dataset eval_data.json --k 5,10,20
```

See [docs/cli_commands.md](docs/cli_commands.md) for the full command reference.

## Architecture

RAG is treated as a four-stage pipeline. These stages are independent modules and must never be collapsed:

```
candidate generation  ──>  precision ranking  ──>  context optimization  ──>  LLM reasoning
     (retrieval)             (reranking)            (context builder)         (generation)
```

The system is organized into three planes:

| Plane | Responsibility | Components |
|-------|---------------|------------|
| **Control** | Configuration and lifecycle | Source registry, ingestion scheduling, experiment definitions, index lifecycle |
| **Data** | Storage and indexing | Raw documents, canonical documents, chunks, embeddings, vector/lexical indexes |
| **Query** | Online request handling | Query normalization, retrieval broker, reranking, context building, generation, tracing |

### Dependency Direction

```
apps/cli/          Thin presentation layer (argparse, output rendering)
  |
orchestrators/     Pipeline composition, trace propagation, degraded-mode logic
  |
libs/*             Business logic, protocols, domain services
  |
libs/contracts     Pure domain types (no logic, no dependencies)
```

Libraries never import each other. Orchestrators wire them together by passing one library's typed output as the next library's input.

## Project Layout

```
apps/                   Application entry points
  cli/                  CLI application (drifter command)
    commands/           Individual command handlers
  api/                  Query-serving HTTP API (planned)
  worker/               Background ingestion worker (planned)
  debugger_ui/          Pipeline debugger UI (planned)

orchestrators/          Pipeline composition
  bootstrap.py          ServiceRegistry + create_registry() composition root
  query.py              QueryOrchestrator (retrieve -> rerank -> context -> generate)
  ingestion.py          IngestionOrchestrator (ingest -> parse -> chunk -> index)

libs/                   Core libraries (one per subsystem)
  contracts/            Shared domain models (typed contracts between subsystems)
  ingestion/            Source discovery, crawl state, document fetching
  parsing/              Format extraction, structure preservation
  chunking/             Document splitting with lineage tracking
  embeddings/           Dense vector generation
  indexing/             Vector and lexical index management
  retrieval/            Query execution
    stores/             VectorStore and LexicalStore protocols + implementations
    broker/             Retrieval orchestration, hybrid fusion (RRF)
  reranking/            Precision ranking of candidates
  context_builder/      Evidence selection under token budgets
  generation/           LLM reasoning with citation grounding
  observability/        OpenTelemetry-compatible tracing, metrics, events
  evaluation/           Retrieval quality metrics (Recall@k, MRR, NDCG)
  experiments/          Reproducible experiment framework
  adapters/             External service integrations + in-memory mocks
    memory/             In-memory implementations for testing
    qdrant/             Qdrant vector store adapter
    opensearch/         OpenSearch vector + lexical adapters
    tei/                Text Embeddings Inference (embeddings, reranking)
    ollama/             Ollama generation, embeddings, query embedding (streaming)
    vllm/               vLLM generation, embeddings, query embedding (streaming)
    openai/             OpenAI generation adapter
    openrouter/         OpenRouter generation + embeddings
    gemini/             Google Gemini generation adapter
    huggingface/        HuggingFace Inference API reranking
    langfuse/           Langfuse observability exporter
    unstructured/       Unstructured PDF parser
    tika/               Apache Tika PDF parser
    ragas/              Ragas evaluation adapter
    otel/               OpenTelemetry span exporter

tests/
  unit/                 Deterministic tests (777 tests, ~0.3s, no external services)
  integration/          Integration tests (planned)
  fixtures/             Sample documents for testing

docs/                   Architecture docs and subsystem guides
prompts/                Numbered implementation prompts (00-16)
experiments/            Experiment workspace organized by subsystem
```

## Domain Types

Typed contracts flow through the pipeline, carrying metadata, lineage, version info, and token counts:

```
SourceDocumentRef -> RawDocument -> CanonicalDocument -> Block -> Chunk
     -> ChunkEmbedding -> RetrievalCandidate -> RankedCandidate
     -> ContextPack -> Citation -> GeneratedAnswer
```

Every type is a frozen dataclass with validation. See [libs/contracts/README.md](libs/contracts/README.md).

## Subsystems

| # | Subsystem | Input | Output | Code |
|---|-----------|-------|--------|------|
| 1 | [Ingestion](libs/ingestion/README.md) | Source configs | SourceDocumentRef, RawDocument | `libs/ingestion/` |
| 2 | [Parsing](libs/parsing/README.md) | RawDocument | CanonicalDocument (with Blocks) | `libs/parsing/` |
| 3 | [Chunking](libs/chunking/README.md) | CanonicalDocument | list[Chunk] with lineage | `libs/chunking/` |
| 4 | [Embeddings](libs/embeddings/README.md) | list[Chunk] | list[ChunkEmbedding] | `libs/embeddings/` |
| 5 | [Indexing](libs/indexing/README.md) | Chunks + Embeddings | IndexingResult | `libs/indexing/` |
| 6 | [Retrieval Stores](libs/retrieval/README.md) | Query + Vector | list[RetrievalCandidate] | `libs/retrieval/stores/` |
| 7 | [Retrieval Broker](libs/retrieval/README.md) | RetrievalQuery | BrokerResult (fused candidates) | `libs/retrieval/broker/` |
| 8 | [Reranking](libs/reranking/README.md) | list[RetrievalCandidate] | list[RankedCandidate] | `libs/reranking/` |
| 9 | [Context Builder](libs/context_builder/README.md) | list[RankedCandidate] | ContextPack | `libs/context_builder/` |
| 10 | [Generation](libs/generation/README.md) | ContextPack | GeneratedAnswer | `libs/generation/` |
| 11 | [Observability](libs/observability/README.md) | (cross-cutting) | Spans, metrics, events | `libs/observability/` |
| 12 | [Evaluation](libs/evaluation/README.md) | EvaluationCase + Retriever | EvaluationReport | `libs/evaluation/` |
| 13 | [Experiments](libs/experiments/README.md) | ExperimentConfig + Retriever | ExperimentRun | `libs/experiments/` |
| 14 | [Adapters](libs/adapters/README.md) | Configs | Concrete service instances | `libs/adapters/` |

## Configuration

Drifter works with zero configuration using in-memory adapters. Production backends activate via environment variables:

| Variable | Service | Default (no env var) |
|----------|---------|---------------------|
| `DRIFTER_QDRANT_HOST` | Qdrant vector store | MemoryVectorStore |
| `DRIFTER_OPENSEARCH_HOSTS` | OpenSearch lexical/vector | MemoryLexicalStore |
| `DRIFTER_TEI_BASE_URL` | Text Embeddings Inference | DeterministicEmbeddingProvider |
| `DRIFTER_OLLAMA_BASE_URL` | Ollama generation | MockGenerator |
| `DRIFTER_VLLM_BASE_URL` | vLLM generation (streaming via SSE) | MockGenerator |
| `DRIFTER_VLLM_EMBEDDINGS_BASE_URL` | vLLM embeddings | DeterministicEmbeddingProvider |
| `DRIFTER_OTEL_ENDPOINT` | OpenTelemetry | NoOpCollector |

```bash
# Copy and edit for your environment
cp .env.example .env

# Start infrastructure services (optional)
docker compose up -d
```

### CLI Overrides

Non-secret config can be overridden at runtime:

```bash
uv run drifter --config token_budget=5000 ask "query"
uv run drifter --config reranker_top_n=10 rerank "query"
uv run drifter --config-file custom_config.yaml ask "query"
uv run drifter --env-file .env.staging ask "query"
```

Secret fields (`api_key`, `password`, `auth`) are rejected from `--config` for security. Use `--config-file` to specify a custom `config.yaml` path and `--env-file` to specify a custom `.env` path.

## Testing

All tests are deterministic and local. No external services required.

```bash
make test          # Run all 777 tests
make test-unit     # Unit tests only
make lint          # Ruff linting
make typecheck     # mypy strict mode
```

Test infrastructure:
- `MemoryVectorStore` / `MemoryLexicalStore` for retrieval
- `DeterministicEmbeddingProvider` / `DeterministicQueryEmbedder` for embeddings
- `MockGenerator` for generation
- `FeatureBasedReranker` for reranking (no external model)
- `InMemoryCollector` for span capture
- In-memory repositories for chunks, embeddings, crawl state

## Evaluation

Changes affecting retrieval quality must include evaluation with metrics:

| Layer | Metrics |
|-------|---------|
| Retrieval | Recall@k, Precision@k, MRR, NDCG@k |
| Answer quality | Faithfulness, citation accuracy, completeness |

```bash
# Evaluate retrieval
uv run drifter evaluate --dataset data.json --k 5,10,20

# Run experiment
uv run drifter experiment run --config experiment.json
```

Evaluate lexical, dense, hybrid, and reranked outputs separately. Recommended datasets: BEIR, HotpotQA, Natural Questions.

## Development

```bash
# Format code
make fmt

# Clean build artifacts
make clean
```

### Design Principles

1. **Design before implementation** - Propose architecture before coding
2. **Strong typed contracts** - Explicit models with metadata, lineage, version info
3. **Deterministic local mode** - Every subsystem testable without external services
4. **Observability by default** - Structured traces/events for all pipeline stages
5. **Evaluation as first-class** - Build retrieval and answer evaluation early
6. **Framework independence** - Core logic decoupled from integration tools

### Key Rules

- Storage-specific code stays behind adapter protocols
- Documents preserve structure (never flatten to plain text)
- Chunks maintain lineage to original documents
- Retrieved documents are untrusted input (prevent prompt injection)
- Generation never fabricates sources
- Experiments are reproducible (record dataset, strategy, model, config, metrics)

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/00_SYSTEM_DESIGN.md](docs/00_SYSTEM_DESIGN.md) | Reference architecture |
| [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) | Directory layout |
| [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) | System structure and boundaries |
| [docs/IMPLEMENTATION_PRINCIPLES.md](docs/IMPLEMENTATION_PRINCIPLES.md) | Design principles |
| [docs/EVALUATION_STRATEGY.md](docs/EVALUATION_STRATEGY.md) | Evaluation layers and metrics |
| [docs/app_layer.md](docs/app_layer.md) | Application layer architecture |
| [docs/cli_commands.md](docs/cli_commands.md) | CLI command reference |
| [docs/developer_workflow.md](docs/developer_workflow.md) | Developer workflow guide |
| [docs/integrations.md](docs/integrations.md) | External service integrations |
| [docs/adding_llm_providers.md](docs/adding_llm_providers.md) | Adding new LLM providers |
