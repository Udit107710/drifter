# Langfuse Integration

Drifter's Langfuse adapter exports pipeline spans as Langfuse traces, observations, and generations. It implements the `SpanCollector` protocol so it plugs into the existing observability infrastructure with zero changes to business logic.

## Quick Start

```bash
# Start Langfuse and its dependencies
docker compose up -d langfuse langfuse-worker

# Configure environment
export DRIFTER_LANGFUSE_PUBLIC_KEY=pk-lf-drifter-dev
export DRIFTER_LANGFUSE_SECRET_KEY=sk-lf-drifter-dev
export DRIFTER_LANGFUSE_HOST=http://localhost:3000
export DRIFTER_LANGFUSE_REDIS_URL=redis://:drifter-redis@localhost:6379

# Run a query — traces appear at http://localhost:3000
uv run rag ask "What is the main theme?"
```

Or just add these variables to `.env` (the CLI auto-loads it).

## Architecture

```
                    ┌──────────────┐
                    │   Tracer     │
                    │  (end-order  │
                    │   export)    │
                    └──────┬───────┘
                           │ collect(span)
                           ▼
                ┌─────────────────────┐
                │ LangfuseSpanExporter│
                │                     │
                │  child span?        │
                │   → buffer.push()   │
                │                     │
                │  root span?         │
                │   → buffer.pop_all()│
                │   → export root     │
                │   → export children │
                │   → flush()         │
                └──────┬──────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │ InMemoryBuf  │  │  RedisBuf    │
    │ (default)    │  │ (production) │
    └──────────────┘  └──────────────┘
```

### Why Buffer?

Drifter's `Tracer` exports spans in **end-order** — child spans finish before their parent. Langfuse names a trace after the first span it receives. Without buffering, a trace might be named "generation" instead of "rag-pipeline".

The exporter solves this by buffering child spans and only flushing when the root span arrives (identified by `parent_span_id=None`). The root is sent first, so Langfuse always names the trace correctly.

### Buffer Backends

| Backend | Config | Use Case |
|---------|--------|----------|
| In-memory | Default (no `redis_url`) | Development, single-process |
| Redis | Set `DRIFTER_LANGFUSE_REDIS_URL` | Production, multi-worker, crash resilience |

Redis keys use the prefix `drifter:langfuse:pending:<trace_id>` and auto-expire after `buffer_ttl_s` (default: 300s) to prevent unbounded growth from orphaned traces.

## What Gets Exported

| Pipeline Stage | Langfuse Type | Metadata |
|---------------|---------------|----------|
| `rag-pipeline` | Trace + root span | query, outcome, duration |
| `retrieval` | Span | input/output counts, store IDs |
| `reranking` | Span | input/output counts, reranker config |
| `context_build` | Span | token budget, evidence count |
| `generation` | **Generation** | model ID, prompt/completion tokens, latency |

Generation observations include `usage_details` with `input` (prompt tokens) and `output` (completion tokens) so Langfuse can compute cost estimates.

## Configuration

| Env Var | Required | Default | Description |
|---------|----------|---------|-------------|
| `DRIFTER_LANGFUSE_PUBLIC_KEY` | Yes | — | Project public key (trigger var) |
| `DRIFTER_LANGFUSE_SECRET_KEY` | Yes | — | Project secret key |
| `DRIFTER_LANGFUSE_HOST` | No | `http://localhost:3000` | Langfuse server URL |
| `DRIFTER_LANGFUSE_REDIS_URL` | No | — | Redis URL for span buffering |
| `DRIFTER_LANGFUSE_BUFFER_TTL_S` | No | `300` | TTL for buffered spans in Redis (seconds) |

When both Langfuse and OTel are configured, Langfuse takes priority.

## Docker Services

The full Langfuse v3 stack requires several services:

```bash
# Start everything
docker compose up -d langfuse langfuse-worker

# This automatically starts dependencies:
#   postgres         — Langfuse metadata store
#   langfuse-clickhouse — Analytics backend
#   langfuse-redis   — Caching + span buffer (shared with Drifter)
#   minio            — S3-compatible blob store
```

| Service | Port | Purpose |
|---------|------|---------|
| `langfuse` | 3000 | Web UI and API |
| `langfuse-worker` | — | Async event processing |
| `langfuse-clickhouse` | — | Analytics storage |
| `langfuse-redis` | 6379 | Caching (Langfuse) + span buffer (Drifter) |
| `postgres` | 5433 | Metadata storage |
| `minio` | 9000/9001 | Event and media blob storage |

**Login**: `admin@drifter.local` / `drifter123!`

## Programmatic Usage

```python
from libs.adapters.config import LangfuseConfig
from libs.adapters.langfuse import LangfuseSpanExporter

config = LangfuseConfig(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="http://localhost:3000",
    redis_url="redis://:password@localhost:6379",
)

exporter = LangfuseSpanExporter(config)
exporter.connect()

# Use as a SpanCollector
tracer = Tracer(collector=exporter)

# ... run pipeline ...

exporter.close()  # flushes remaining spans
```

Or use the factory:

```python
from libs.adapters.factory import create_span_collector
from libs.adapters.env import load_langfuse_config

collector = create_span_collector(load_langfuse_config())
```

## File Structure

```
libs/adapters/langfuse/
├── __init__.py      # Package exports
├── exporter.py      # LangfuseSpanExporter, buffer implementations
└── README.md        # This file
```

## Troubleshooting

**Traces named "generation" instead of "rag-pipeline"**: The buffer isn't working. Check that the root pipeline span has `parent_span_id=None`. If using Redis, verify connectivity with `redis-cli -a drifter-redis ping`.

**Missing generation token counts**: Some models (e.g. Gemma) don't report completion token counts. The Gemini adapter falls back to word-count estimates.

**Langfuse UI shows no data**: The `langfuse-worker` container must be running — it processes events from the queue. Check `docker compose logs langfuse-worker`.

**Redis connection refused**: Ensure `langfuse-redis` is running and port 6379 is exposed. The docker-compose maps it to the host.
