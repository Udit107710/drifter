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
DRIFTER_LANGFUSE_PUBLIC_KEY=pk-lf-drifter-dev
DRIFTER_LANGFUSE_SECRET_KEY=sk-lf-drifter-dev
DRIFTER_LANGFUSE_HOST=http://localhost:3000
DRIFTER_LANGFUSE_REDIS_URL=redis://:drifter-redis@localhost:6379
```

### 2. Ingest data

```bash
# Ingest a directory of markdown files
uv run drifter ingest --path data/novel_chapters/

# Ingest a single file
uv run drifter ingest --path notes.md
```

### 3. Query

```bash
# Ask a question (full pipeline: retrieve → rerank → context → generate)
uv run drifter ask "What is the main theme?"

# JSON output
uv run drifter --json ask "What is the main theme?"

# Debug mode (full pipeline JSON with all intermediate results)
uv run drifter debug-query "What is the main theme?"
```

Without external services configured, the system uses in-memory stores and a mock generator.

## Docker Services

Start all infrastructure services:

```bash
# Core storage (Qdrant + OpenSearch)
docker compose up -d qdrant opensearch

# Langfuse observability (includes ClickHouse, Redis, Postgres, MinIO)
docker compose up -d langfuse langfuse-worker
```

### Local Model Serving

Generation, embeddings, and reranking run outside docker compose. Three local-first options:

#### vLLM — Generation and Embeddings (GPU)

vLLM serves OpenAI-compatible endpoints for both generation and embeddings:

```bash
# vLLM generation (thinking model)
vllm serve Qwen/Qwen3-8B-AWQ --port 8000 --gpu-memory-utilization 0.85 --max-model-len 32768 --reasoning-parser qwen3

# vLLM embeddings
vllm serve nomic-ai/nomic-embed-text-v1.5 --port 8001 --gpu-memory-utilization 0.10 --max-model-len 2048 --convert embed
```

Configure in `config.yaml`:

```yaml
generation:
  provider: vllm
  vllm_url: http://localhost:8000

embeddings:
  provider: vllm
  vllm_url: http://localhost:8001
```

#### TEI — Embeddings and Reranking (GPU)

TEI runs as standalone docker containers (not part of docker compose):

```bash
# TEI embeddings on GPU
docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest --model-id nomic-ai/nomic-embed-text-v1.5

# TEI reranker on GPU
docker run --gpus all -p 8081:80 ghcr.io/huggingface/text-embeddings-inference:latest --model-id BAAI/bge-reranker-v2-m3
```

```bash
# Verify health
curl http://localhost:8080/health
curl http://localhost:8081/health
```

Configure in `.env`:

```env
DRIFTER_TEI_URL=http://localhost:8080
DRIFTER_TEI_RERANKER_URL=http://localhost:8081
```

When `DRIFTER_TEI_RERANKER_URL` is set and the server is reachable, the bootstrap uses `TeiCrossEncoderReranker` instead of the local `FeatureBasedReranker`. When not set or unreachable, it falls back automatically.

#### Local CPU Reranker — No External Service

The `FeatureBasedReranker` uses transformers locally and requires no external service. It is the default when no TEI reranker URL is configured. Set `reranking.provider: feature` in `config.yaml` to use it explicitly.

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

The recommended approach is to use `config.yaml` for non-secret configuration and `.env` for secrets only. See `config.yaml.example` for the full schema. When no `config.yaml` exists, the system falls back to `DRIFTER_*` env vars. The CLI auto-loads a `.env` file from the project root, so you don't need to export variables manually.

| Variable | Service |
|----------|---------|
| `DRIFTER_QDRANT_HOST` | Qdrant vector store |
| `DRIFTER_OPENSEARCH_HOSTS` | OpenSearch lexical/vector store |
| `DRIFTER_TEI_URL` | Text Embeddings Inference (embeddings) |
| `DRIFTER_TEI_RERANKER_URL` | TEI cross-encoder reranking |
| `DRIFTER_VLLM_URL` | vLLM generation endpoint |
| `DRIFTER_VLLM_EMBEDDINGS_URL` | vLLM embeddings endpoint |
| `DRIFTER_OPENROUTER_API_KEY` | OpenRouter LLM gateway |
| `DRIFTER_OPENAI_API_KEY` | OpenAI LLM generation |
| `DRIFTER_GEMINI_API_KEY` | Google Gemini LLM generation |
| `DRIFTER_OLLAMA_URL` | Ollama generation and embeddings |
| `DRIFTER_OTEL_ENDPOINT` | OpenTelemetry collector |
| `DRIFTER_LANGFUSE_PUBLIC_KEY` | Langfuse observability |
| `DRIFTER_LANGFUSE_REDIS_URL` | Redis buffer for Langfuse span export |

When using env-var fallback (no `config.yaml`), LLM provider priority is: OpenRouter > OpenAI > Gemini > Ollama.
With `config.yaml`, set `generation.provider` explicitly (e.g., `ollama`, `vllm`, `openai`, `openrouter`, `gemini`).
When both `DRIFTER_LANGFUSE_PUBLIC_KEY` and `DRIFTER_OTEL_ENDPOINT` are set, Langfuse is preferred.

### CLI Overrides

Non-secret config can be overridden at runtime:

```bash
uv run drifter --config token_budget=5000 ask "query"
uv run drifter --config reranker_top_n=10 rerank "query"
```

Secret fields (`api_key`, `password`, `auth`, `secret`) cannot be set via `--config` for security.

Use `--config-file` and `--env-file` to point at non-default config and env files:

```bash
uv run drifter --config-file /path/to/config.yaml ask "query"
uv run drifter --env-file /path/to/.env ask "query"
```

## Pipeline Stages

### Ingestion (write path)

```bash
# Ingest documents from a directory
uv run drifter ingest --path data/novel_chapters/

# Ingest a single file with explicit run ID
uv run drifter ingest --path document.md --run-id run-001
```

The ingestion pipeline: read files → parse (Markdown/plain text) → chunk (recursive structure) → embed → store in Qdrant (vector) + OpenSearch (lexical).

### Query (read path)

Each stage can be run independently:

```bash
# Retrieval only
uv run drifter retrieve "query" --mode dense --top-k 10

# Retrieval + reranking
uv run drifter rerank "query" --top-k 10

# Retrieval + reranking + context
uv run drifter build-context "query" --token-budget 2000

# Full pipeline
uv run drifter ask "query"
```

## Evaluation

```bash
# Run evaluation against a dataset
uv run drifter evaluate --dataset eval_data.json --k 5,10,20

# Run experiment
uv run drifter experiment run --config experiment.json --hypothesis "Hybrid beats dense"
```

## Trace IDs

Every command generates a trace ID (visible in stderr). Use `--trace` to set a specific one:

```bash
uv run drifter --trace my-debug-123 ask "query"
```

Trace IDs propagate through all pipeline stages and appear in all observability spans.

## Langfuse Observability

Langfuse provides LLM-focused observability with traces, spans, and generation tracking.

### Setup

```bash
# Start Langfuse and its dependencies (ClickHouse, Redis, Postgres, MinIO)
docker compose up -d langfuse langfuse-worker

# Langfuse UI: http://localhost:3000
# Login: admin@drifter.local / drifter123!
```

### What's Tracked

- **Traces**: One per pipeline execution, grouped by trace_id
- **Spans**: Retrieval, reranking, and context building stages
- **Generations**: LLM calls with model name, token usage (prompt/completion), and latency
- **Parent-child relationships**: Pipeline stages appear as nested observations

### Configuration

```env
DRIFTER_LANGFUSE_PUBLIC_KEY=pk-lf-drifter-dev
DRIFTER_LANGFUSE_SECRET_KEY=sk-lf-drifter-dev
DRIFTER_LANGFUSE_HOST=http://localhost:3000
DRIFTER_LANGFUSE_REDIS_URL=redis://:drifter-redis@localhost:6379
DRIFTER_LANGFUSE_BUFFER_TTL_S=300
```

### Span Buffering

The exporter buffers child spans until the root pipeline span arrives, ensuring traces are named correctly. Two backends are available:

- **In-memory** (default): No extra config needed. Simple and fast but spans are lost on crash.
- **Redis**: Set `DRIFTER_LANGFUSE_REDIS_URL` to enable. Survives restarts, works across multiple workers. Buffered spans auto-expire after `DRIFTER_LANGFUSE_BUFFER_TTL_S` seconds (default: 300).

The local docker-compose exposes `langfuse-redis` on port 6379 — the same Redis instance used by Langfuse itself.

When Langfuse is configured, it takes priority over OpenTelemetry (Jaeger).
