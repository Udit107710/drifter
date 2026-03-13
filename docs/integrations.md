# Integration Adapters

Drifter uses adapters under `libs/adapters/` to isolate external provider dependencies behind protocol boundaries. Most adapters are fully implemented with real service backends; a few remain as stubs. Each adapter is importable without its underlying client library installed.

## Provider Map

| Provider | Adapter | Protocol / Base |
|----------|---------|-----------------|
| Qdrant | `libs.adapters.qdrant.QdrantVectorStore` | `VectorStore` |
| OpenSearch (k-NN) | `libs.adapters.opensearch.OpenSearchVectorStore` | `VectorStore` |
| OpenSearch (BM25) | `libs.adapters.opensearch.OpenSearchLexicalStore` | `LexicalStore` |
| TEI (embeddings) | `libs.adapters.tei.TeiEmbeddingProvider` | `EmbeddingProvider` |
| TEI (query) | `libs.adapters.tei.TeiQueryEmbedder` | `QueryEmbedder` |
| TEI (reranking) | `libs.adapters.tei.TeiCrossEncoderReranker` | `Reranker` |
| HuggingFace (reranking) | `libs.adapters.huggingface.HuggingFaceReranker` | `Reranker` |
| OpenRouter (generation) | `libs.adapters.openai.OpenAIGenerator` (via factory) | `Generator` |
| OpenRouter (embeddings) | `libs.adapters.openrouter.OpenRouterEmbeddingProvider` | `EmbeddingProvider` |
| OpenRouter (query) | `libs.adapters.openrouter.OpenRouterQueryEmbedder` | `QueryEmbedder` |
| OpenAI | `libs.adapters.openai.OpenAIGenerator` | `Generator` |
| Google Gemini | `libs.adapters.gemini.GeminiGenerator` | `Generator` |
| Ollama (generation) | `libs.adapters.ollama.OllamaGenerator` | `Generator` |
| Ollama (embeddings) | `libs.adapters.ollama.OllamaEmbeddingProvider` | `EmbeddingProvider` |
| Ollama (query) | `libs.adapters.ollama.OllamaQueryEmbedder` | `QueryEmbedder` |
| vLLM (generation) | `libs.adapters.vllm.VllmGenerator` | `Generator` |
| vLLM (embeddings) | `libs.adapters.vllm.VllmEmbeddingProvider` | `EmbeddingProvider` |
| vLLM (query) | `libs.adapters.vllm.VllmQueryEmbedder` | `QueryEmbedder` |
| Unstructured | `libs.adapters.unstructured.UnstructuredPdfParser` | `PdfParserBase` |
| Apache Tika | `libs.adapters.tika.TikaPdfParser` | `PdfParserBase` |
| Ragas | `libs.adapters.ragas.RagasAnswerEvaluator` | (standalone) |
| OpenTelemetry | `libs.adapters.otel.OtelSpanExporter` | `SpanCollector` |
| Local CPU (reranking) | `libs.reranking.local_cross_encoder.LocalCrossEncoderReranker` | `Reranker` |
| Langfuse | `libs.adapters.langfuse.LangfuseSpanExporter` | `SpanCollector` |

## Configuration

All configs live in `libs/adapters/config.py` as frozen dataclasses. Secret fields (`api_key`, `password`) are masked in `__repr__`.

### Config YAML (recommended)

Non-secret settings go in `config.yaml` at the project root. Explicit `provider` fields select which adapter to use:

```yaml
generation:
  provider: vllm            # ollama | vllm | openai | openrouter | gemini
embeddings:
  provider: vllm            # tei | ollama | vllm | openrouter
reranking:
  provider: local           # local | tei | huggingface | feature
observability:
  provider: langfuse        # langfuse | otel
```

Secrets are injected from environment variables automatically (e.g., `DRIFTER_OPENAI_API_KEY`).

### Environment Variables (fallback)

When no `config.yaml` exists, set `DRIFTER_*` env vars to configure providers. The `libs.adapters.env` module provides `load_*_config()` functions that return `None` when the primary env var is absent.

| Provider | Primary Env Var | Additional Vars |
|----------|----------------|-----------------|
| Qdrant | `DRIFTER_QDRANT_HOST` | `_PORT`, `_GRPC_PORT`, `_API_KEY`, `_COLLECTION`, `_TIMEOUT_S`, `_USE_TLS` |
| OpenSearch | `DRIFTER_OPENSEARCH_HOSTS` | `_USERNAME`, `_PASSWORD`, `_INDEX_PREFIX`, `_USE_SSL`, `_TIMEOUT_S` |
| OpenRouter | `DRIFTER_OPENROUTER_API_KEY` | `_MODEL`, `_EMBEDDING_MODEL`, `_BASE_URL`, `_APP_NAME`, `_TIMEOUT_S`, `_MAX_TOKENS`, `_MAX_BATCH_SIZE`, `_TEMPERATURE` |
| OpenAI | `DRIFTER_OPENAI_API_KEY` | `_MODEL` (default: gpt-4o), `_BASE_URL`, `_TIMEOUT_S`, `_MAX_TOKENS`, `_TEMPERATURE` |
| TEI | `DRIFTER_TEI_URL` | `_RERANKER_URL`, `_MODEL_ID`, `_MODEL_VERSION`, `_RERANKER_MODEL_ID`, `_TIMEOUT_S`, `_MAX_BATCH_SIZE` |
| HuggingFace | `DRIFTER_HF_TOKEN` | `_RERANKER_MODEL`, `_PROVIDER`, `_TIMEOUT_S` |
| Gemini | `DRIFTER_GEMINI_API_KEY` | `_MODEL` (default: gemini-2.5-flash), `_TIMEOUT_S`, `_MAX_TOKENS`, `_TEMPERATURE` |
| Ollama | `DRIFTER_OLLAMA_URL` | `_MODEL_ID`, `_TIMEOUT_S`, `_NUM_PREDICT`, `_NUM_CTX`, `_TEMPERATURE`, `_TOP_K`, `_TOP_P`, `_MIN_P`, `_SEED`, `_REPEAT_PENALTY`, `_REPEAT_LAST_N`, `_STOP`, `_KEEP_ALIVE` |
| Unstructured | `DRIFTER_UNSTRUCTURED_URL` | `_STRATEGY`, `_TIMEOUT_S` |
| Tika | `DRIFTER_TIKA_URL` | `_TIMEOUT_S` |
| Ragas | `DRIFTER_RAGAS_MODEL` | `_METRICS` (comma-separated) |
| OTel | `DRIFTER_OTEL_ENDPOINT` | `_PROTOCOL`, `_SERVICE_NAME`, `_EXPORT_INTERVAL_MS`, `_INSECURE` |
| Langfuse | `DRIFTER_LANGFUSE_PUBLIC_KEY` | `_SECRET_KEY`, `_HOST` (default: `http://localhost:3000`), `_REDIS_URL`, `_BUFFER_TTL_S` (default: 300) |

## Factory Functions

Use `libs.adapters.factory` to create adapter instances:

```python
from libs.adapters.factory import create_vector_store, create_generator
from libs.adapters.env import load_qdrant_config, load_ollama_config

vector_store = create_vector_store(load_qdrant_config())  # MemoryVectorStore if unset
generator = create_generator(load_ollama_config())        # MockGenerator if unset
```

Each factory returns an in-memory/mock fallback when `config=None`, keeping the system fully functional without external services.

## Factory Return Types

All factory functions return properly typed Protocol instances:

| Factory | Return Type |
|---------|-------------|
| `create_vector_store()` | `VectorStore` |
| `create_lexical_store()` | `LexicalStore` |
| `create_embedding_provider()` | `EmbeddingProvider` |
| `create_query_embedder()` | `QueryEmbedder` |
| `create_reranker()` | `Reranker` |
| `create_generator()` | `Generator` |
| `create_span_collector()` | `SpanCollector` |
| `create_pdf_parser()` | `PdfParserBase` |

This enables static type checking (mypy) at all call sites.

## Lifecycle

Adapter lifecycle is managed through two runtime-checkable protocols defined in `libs/adapters/protocols.py`:

- **`Connectable`** — adapters that require an explicit connection step expose `connect() -> None`
- **`HealthCheckable`** — adapters that support health checks expose `health_check() -> bool`

All real adapters additionally expose `close()` for resource cleanup.

The bootstrap (`orchestrators/bootstrap.py`) uses `isinstance(obj, Connectable)` and `isinstance(obj, HealthCheckable)` to safely call lifecycle methods only on adapters that support them — no duck-typing or `hasattr` checks.

## Implemented Adapters

The following adapters are fully implemented with real service backends:

| Adapter | Client | Notes |
|---------|--------|-------|
| `QdrantVectorStore` | `qdrant-client` | gRPC or REST, TLS optional |
| `OpenSearchVectorStore` | `opensearch-py` | k-NN plugin required |
| `OpenSearchLexicalStore` | `opensearch-py` | BM25 full-text search |
| `OtelSpanExporter` | `opentelemetry-sdk` | OTLP gRPC/HTTP export |
| `LangfuseSpanExporter` | `langfuse` | Redis-backed span buffering |
| `OpenAIGenerator` (OpenRouter) | `httpx` | OpenRouter LLM generation |
| `OpenRouterEmbeddingProvider` | `httpx` | OpenRouter embeddings (batch) |
| `OpenRouterQueryEmbedder` | `httpx` | OpenRouter query embeddings |
| `OpenAIGenerator` | `httpx` | OpenAI Chat Completions API |
| `GeminiGenerator` | `google-genai` | Gemini 2.5 Flash/Pro |
| `TeiEmbeddingProvider` | `httpx` | TEI `/embed` + `/info` endpoints, batched |
| `TeiQueryEmbedder` | `httpx` | TEI `/embed` for single queries |
| `TeiCrossEncoderReranker` | `httpx` | TEI `/rerank` endpoint |
| `HuggingFaceReranker` | `huggingface_hub` | HF Inference API cross-encoder reranking |
| `OllamaGenerator` | `httpx` | Ollama local LLM generation (default port 11434) |
| `OllamaEmbeddingProvider` | `httpx` | Ollama embeddings (batch via /api/embed) |
| `OllamaQueryEmbedder` | `httpx` | Ollama single-query embedding |
| `VllmGenerator` | `httpx` | vLLM Chat Completions (SSE streaming, thinking models) |
| `VllmEmbeddingProvider` | `httpx` | vLLM /v1/embeddings (batch) |
| `VllmQueryEmbedder` | `httpx` | vLLM single-query embedding |
| `LocalCrossEncoderReranker` | `transformers` + `torch` | CPU cross-encoder, lazy model load |

## Stub Adapters (Not Yet Implemented)

| Adapter | Status |
|---------|--------|
| `UnstructuredPdfParser` | Stub — raises `NotImplementedError` |
| `TikaPdfParser` | Stub — raises `NotImplementedError` |
| `RagasAnswerEvaluator` | Stub — raises `NotImplementedError` |

## Adding a Real Implementation

1. Install the provider's client library
2. Implement the data methods in the corresponding adapter file
3. Wire `connect()`/`close()` to manage the client lifecycle
4. Update `health_check()` to ping the service
5. Run `uv run pytest tests/` to verify protocol compliance

See `examples/integration_wiring_sample.py` for a complete wiring example.
