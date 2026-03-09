# Integration Adapters

Drifter uses adapter stubs under `libs/adapters/` to isolate external provider dependencies behind protocol boundaries. Each adapter is importable without its underlying client library installed.

## Provider Map

| Provider | Adapter | Protocol / Base |
|----------|---------|-----------------|
| Qdrant | `libs.adapters.qdrant.QdrantVectorStore` | `VectorStore` |
| OpenSearch (k-NN) | `libs.adapters.opensearch.OpenSearchVectorStore` | `VectorStore` |
| OpenSearch (BM25) | `libs.adapters.opensearch.OpenSearchLexicalStore` | `LexicalStore` |
| TEI (embeddings) | `libs.adapters.tei.TeiEmbeddingProvider` | `EmbeddingProvider` |
| TEI (query) | `libs.adapters.tei.TeiQueryEmbedder` | `QueryEmbedder` |
| TEI (reranking) | `libs.adapters.tei.TeiCrossEncoderReranker` | `Reranker` |
| Google Gemini | `libs.adapters.gemini.GeminiGenerator` | `Generator` |
| vLLM | `libs.adapters.vllm.VllmGenerator` | `Generator` |
| Unstructured | `libs.adapters.unstructured.UnstructuredPdfParser` | `PdfParserBase` |
| Apache Tika | `libs.adapters.tika.TikaPdfParser` | `PdfParserBase` |
| Ragas | `libs.adapters.ragas.RagasAnswerEvaluator` | (standalone) |
| OpenTelemetry | `libs.adapters.otel.OtelSpanExporter` | `SpanCollector` |
| Langfuse | `libs.adapters.langfuse.LangfuseSpanExporter` | `SpanCollector` |

## Configuration

All configs live in `libs/adapters/config.py` as frozen dataclasses. Secret fields (`api_key`, `password`) are masked in `__repr__`.

### Environment Variables

Set `DRIFTER_*` env vars to configure providers. The `libs.adapters.env` module provides `load_*_config()` functions that return `None` when the primary env var is absent.

| Provider | Primary Env Var | Additional Vars |
|----------|----------------|-----------------|
| Qdrant | `DRIFTER_QDRANT_HOST` | `_PORT`, `_GRPC_PORT`, `_API_KEY`, `_COLLECTION`, `_TIMEOUT_S`, `_USE_TLS` |
| OpenSearch | `DRIFTER_OPENSEARCH_HOSTS` | `_USERNAME`, `_PASSWORD`, `_INDEX_PREFIX`, `_USE_SSL`, `_TIMEOUT_S` |
| TEI | `DRIFTER_TEI_URL` | `_MODEL_ID`, `_MODEL_VERSION`, `_TIMEOUT_S`, `_MAX_BATCH_SIZE` |
| Gemini | `DRIFTER_GEMINI_API_KEY` | `_MODEL` (default: gemini-2.5-flash), `_TIMEOUT_S`, `_MAX_TOKENS`, `_TEMPERATURE` |
| vLLM | `DRIFTER_VLLM_URL` | `_MODEL_ID`, `_API_KEY`, `_TIMEOUT_S`, `_MAX_TOKENS`, `_TEMPERATURE` |
| Unstructured | `DRIFTER_UNSTRUCTURED_URL` | `_STRATEGY`, `_TIMEOUT_S` |
| Tika | `DRIFTER_TIKA_URL` | `_TIMEOUT_S` |
| Ragas | `DRIFTER_RAGAS_MODEL` | `_METRICS` (comma-separated) |
| OTel | `DRIFTER_OTEL_ENDPOINT` | `_PROTOCOL`, `_SERVICE_NAME`, `_EXPORT_INTERVAL_MS`, `_INSECURE` |
| Langfuse | `DRIFTER_LANGFUSE_PUBLIC_KEY` | `_SECRET_KEY`, `_HOST` (default: `http://localhost:3000`), `_REDIS_URL`, `_BUFFER_TTL_S` (default: 300) |

## Factory Functions

Use `libs.adapters.factory` to create adapter instances:

```python
from libs.adapters.factory import create_vector_store, create_generator
from libs.adapters.env import load_qdrant_config, load_vllm_config

vector_store = create_vector_store(load_qdrant_config())  # MemoryVectorStore if unset
generator = create_generator(load_vllm_config())          # MockGenerator if unset
```

Each factory returns an in-memory/mock fallback when `config=None`, keeping the system fully functional without external services.

## Lifecycle

Every adapter stub exposes three lifecycle methods:

- `connect()` — establish connection (no-op in stubs)
- `close()` — release resources (no-op in stubs)
- `health_check() -> bool` — returns `False` in stubs

## Adding a Real Implementation

1. Install the provider's client library
2. Implement the data methods in the corresponding adapter file
3. Wire `connect()`/`close()` to manage the client lifecycle
4. Update `health_check()` to ping the service
5. Run `uv run pytest tests/` to verify protocol compliance

See `examples/integration_wiring_sample.py` for a complete wiring example.
