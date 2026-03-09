# Developer Workflow

## Getting Started

The system works with zero external services using in-memory/mock implementations.

### 1. Configure environment

Copy `.env.example` to `.env` and fill in values for the services you want to use. The CLI auto-loads `.env` from the project root on startup — no need to export variables manually.

```bash
cp .env.example .env
# Edit .env with your values
```

Minimal `.env` for local development with Docker services:

```env
DRIFTER_QDRANT_HOST=localhost
DRIFTER_OPENSEARCH_HOSTS=localhost:9200
DRIFTER_OPENSEARCH_USERNAME=admin
DRIFTER_OPENSEARCH_PASSWORD=admin
DRIFTER_OPENSEARCH_USE_SSL=false
DRIFTER_GEMINI_API_KEY=your-api-key-here
DRIFTER_OTEL_ENDPOINT=http://localhost:4318
```

### 2. Ingest data

```bash
# Ingest a directory of markdown files
uv run rag ingest --path data/novel_chapters/

# Ingest a single file
uv run rag ingest --path notes.md
```

### 3. Query

```bash
# Ask a question (full pipeline: retrieve → rerank → context → generate)
uv run rag ask "What is the main theme?"

# JSON output
uv run rag --json ask "What is the main theme?"

# Debug mode (full pipeline JSON with all intermediate results)
uv run rag debug-query "What is the main theme?"
```

Without external services configured, the system uses in-memory stores and a mock generator.

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/unit/test_query_orchestrator.py -v

# Lint
uv run ruff check .

# Type check
uv run mypy orchestrators/ apps/cli/ --strict
```

## Configuration

### Environment Variables

All external services are configured via `DRIFTER_*` env vars. When not set, in-memory fallbacks are used. The CLI auto-loads a `.env` file from the project root, so you don't need to export these manually.

| Variable | Service |
|----------|---------|
| `DRIFTER_QDRANT_HOST` | Qdrant vector store |
| `DRIFTER_OPENSEARCH_HOSTS` | OpenSearch lexical/vector store |
| `DRIFTER_TEI_URL` | Text Embeddings Inference |
| `DRIFTER_GEMINI_API_KEY` | Google Gemini LLM generation |
| `DRIFTER_VLLM_URL` | vLLM generation |
| `DRIFTER_OTEL_ENDPOINT` | OpenTelemetry collector |

When both `DRIFTER_GEMINI_API_KEY` and `DRIFTER_VLLM_URL` are set, Gemini is preferred.

### CLI Overrides

Non-secret config can be overridden at runtime:

```bash
uv run rag --config token_budget=5000 ask "query"
uv run rag --config reranker_top_n=10 rerank "query"
```

Secret fields (`api_key`, `password`, `auth`, `secret`) cannot be set via `--config` for security.

## Pipeline Stages

### Ingestion (write path)

```bash
# Ingest documents from a directory
uv run rag ingest --path data/novel_chapters/

# Ingest a single file with explicit run ID
uv run rag ingest --path document.md --run-id run-001
```

The ingestion pipeline: read files → parse (Markdown/plain text) → chunk (recursive structure) → embed → store in Qdrant (vector) + OpenSearch (lexical).

### Query (read path)

Each stage can be run independently:

```bash
# Retrieval only
uv run rag retrieve "query" --mode dense --top-k 10

# Retrieval + reranking
uv run rag rerank "query" --top-k 10

# Retrieval + reranking + context
uv run rag build-context "query" --token-budget 2000

# Full pipeline
uv run rag ask "query"
```

## Evaluation

```bash
# Run evaluation against a dataset
uv run rag evaluate --dataset eval_data.json --k 5,10,20

# Run experiment
uv run rag experiment run --config experiment.json --hypothesis "Hybrid beats dense"
```

## Trace IDs

Every command generates a trace ID (visible in stderr). Use `--trace` to set a specific one:

```bash
uv run rag --trace my-debug-123 ask "query"
```

Trace IDs propagate through all pipeline stages and appear in all observability spans.
