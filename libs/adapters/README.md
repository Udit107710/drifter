# libs/adapters/

Concrete implementations of subsystem protocols for external services, plus in-memory mocks for testing.

## Architecture

Every external dependency is behind a protocol defined in its subsystem. Adapters implement these protocols. Factory functions in `factory.py` select the right implementation based on configuration.

```
libs/adapters/factory.py       # Factory functions (create_vector_store, create_generator, ...)
libs/adapters/config.py        # Config dataclasses (QdrantConfig, OllamaConfig, VllmConfig, ...)
libs/adapters/config_loader.py # YAML config loader with DrifterConfig and explicit provider selection
libs/adapters/env.py           # Environment loaders (load_qdrant_config, ...) — fallback mode
libs/adapters/memory/          # In-memory implementations
libs/adapters/qdrant/          # Qdrant adapter
libs/adapters/opensearch/      # OpenSearch adapter
libs/adapters/tei/             # TEI adapter (embeddings, query embedding, reranking)
libs/adapters/ollama/          # Ollama adapter (generation, embeddings, query embedding)
libs/adapters/vllm/            # vLLM adapter (generation, embeddings, query embedding)
libs/adapters/openai/          # OpenAI generation adapter
libs/adapters/openrouter/      # OpenRouter generation + embeddings adapter
libs/adapters/gemini/          # Google Gemini generation adapter
libs/adapters/huggingface/     # HuggingFace reranking adapter
libs/adapters/langfuse/        # Langfuse observability exporter
libs/adapters/unstructured/    # Unstructured adapter
libs/adapters/tika/            # Tika adapter
libs/adapters/ragas/           # Ragas adapter
libs/adapters/otel/            # OpenTelemetry adapter
```

## Factory Functions (`factory.py`)

Each factory returns an in-memory/mock implementation when `config=None`, and the real adapter when config is provided:

| Factory | Protocol | Mock Fallback | Real Adapter |
|---------|----------|---------------|--------------|
| `create_vector_store(config)` | `VectorStore` | `MemoryVectorStore` | `QdrantVectorStore` / `OpenSearchVectorStore` |
| `create_lexical_store(config)` | `LexicalStore` | `MemoryLexicalStore` | `OpenSearchLexicalStore` |
| `create_embedding_provider(config)` | `EmbeddingProvider` | `DeterministicEmbeddingProvider` | `TeiEmbeddingProvider`, `OllamaEmbeddingProvider`, `VllmEmbeddingProvider`, `OpenRouterEmbeddingProvider` |
| `create_query_embedder(config)` | `QueryEmbedder` | Mock hash-based embedder | `TeiQueryEmbedder`, `OllamaQueryEmbedder`, `VllmQueryEmbedder`, `OpenRouterQueryEmbedder` |
| `create_reranker(config)` | `Reranker` | `CrossEncoderReranker` (stub) | `TeiCrossEncoderReranker`, `HuggingFaceReranker` |
| `create_generator(config)` | `Generator` | `MockGenerator` | `OllamaGenerator`, `VllmGenerator`, `OpenAIGenerator`, `GeminiGenerator` |
| `create_span_collector(config)` | `SpanCollector` | `NoOpCollector` | `OtelSpanExporter`, `LangfuseSpanExporter` |
| `create_pdf_parser(provider, config)` | `PdfParserBase` | — | `UnstructuredPdfParser` / `TikaPdfParser` |

## Environment Loaders (`env.py`)

Each loader reads `DRIFTER_*` environment variables and returns a config dataclass or `None` if the required vars aren't set:

| Loader | Env Vars | Config Type |
|--------|----------|-------------|
| `load_qdrant_config()` | `DRIFTER_QDRANT_*` | `QdrantConfig` |
| `load_opensearch_config()` | `DRIFTER_OPENSEARCH_*` | `OpenSearchConfig` |
| `load_tei_config()` | `DRIFTER_TEI_*` | `TeiConfig` |
| `load_ollama_config()` | `DRIFTER_OLLAMA_*` | `OllamaConfig` |
| `load_otel_config()` | `DRIFTER_OTEL_*` | `OtelConfig` |
| `load_unstructured_config()` | `DRIFTER_UNSTRUCTURED_*` | `UnstructuredConfig` |
| `load_tika_config()` | `DRIFTER_TIKA_*` | `TikaConfig` |
| `load_ragas_config()` | `DRIFTER_RAGAS_*` | `RagasConfig` |

## Config Dataclasses (`config.py`)

All configs are `@dataclass(frozen=True)` with validation and sensible defaults:

| Config | Key Fields |
|--------|-----------|
| `QdrantConfig` | host, port, api_key, collection_name, use_tls |
| `OpenSearchConfig` | hosts, username, password, index_prefix, use_ssl |
| `TeiConfig` | base_url, model_id, model_version, max_batch_size |
| `OllamaConfig` | base_url, model_id, timeout_s, num_predict, num_ctx, temperature, top_k, top_p, min_p, seed, repeat_penalty, repeat_last_n, stop, keep_alive |
| `VllmConfig` | base_url, model_id, timeout_s, max_tokens, temperature, top_k, top_p, min_p, repetition_penalty, stop |
| `OpenAIConfig` | api_key, model_id, base_url, timeout_s, max_tokens, temperature |
| `OpenRouterConfig` | api_key, model_id, embedding_model, base_url, app_name, timeout_s, max_tokens, max_batch_size, temperature |
| `GeminiConfig` | api_key, model_id, timeout_s, max_tokens, temperature |
| `HuggingFaceConfig` | api_key, reranker_model, provider, timeout_s |
| `LangfuseConfig` | public_key, secret_key, host, redis_url, buffer_ttl_s |
| `OtelConfig` | endpoint, protocol, service_name, insecure |
| `UnstructuredConfig` | base_url, strategy |
| `TikaConfig` | base_url |
| `RagasConfig` | model_id, metrics |

Secret fields use `_masked_repr()` to prevent credential logging.

## In-Memory Adapters (`memory/`)

Implementations for deterministic local testing:

| Adapter | Protocol |
|---------|----------|
| `MemorySourceRepository` | `SourceRepository` |
| `MemoryCrawlStateRepository` | `CrawlStateRepository` |
| `MemoryChunkRepository` | `ChunkRepository` |
| `MemoryEmbeddingRepository` | `EmbeddingRepository` |
| `MemoryVectorIndexWriter` | `VectorIndexWriter` |
| `MemoryLexicalIndexWriter` | `LexicalIndexWriter` |

## Adapter Modules

### Qdrant (`qdrant/`)
- `QdrantVectorStore` — Qdrant vector store implementing `VectorStore` protocol

### OpenSearch (`opensearch/`)
- `OpenSearchVectorStore` — OpenSearch vector store implementing `VectorStore`
- `OpenSearchLexicalStore` — OpenSearch lexical store implementing `LexicalStore`

### TEI (`tei/`)
- `TeiEmbeddingProvider` — Embeddings via Text Embeddings Inference
- `TeiQueryEmbedder` — Query embedding via TEI
- `TeiCrossEncoderReranker` — Cross-encoder reranking via TEI

### Ollama (`ollama/`)
- `OllamaGenerator` — LLM generation via native `/api/chat` endpoint (supports streaming via NDJSON)
- `OllamaEmbeddingProvider` — Embeddings via native `/api/embed` endpoint
- `OllamaQueryEmbedder` — Query embedding via `/api/embed`

### vLLM (`vllm/`)
- `VllmGenerator` — LLM generation via `/v1/chat/completions` with native params (supports streaming via SSE)
- `VllmEmbeddingProvider` — Embeddings via `/v1/embeddings`
- `VllmQueryEmbedder` — Query embedding via `/v1/embeddings`

### OpenAI (`openai/`)
- `OpenAIGenerator` — LLM generation via OpenAI Chat Completions API

### OpenRouter (`openrouter/`)
- `OpenRouterEmbeddingProvider` — Embeddings via OpenRouter API
- `OpenRouterQueryEmbedder` — Query embedding via OpenRouter API

### Gemini (`gemini/`)
- `GeminiGenerator` — LLM generation via Google Gemini API

### HuggingFace (`huggingface/`)
- `HuggingFaceReranker` — Cross-encoder reranking via HuggingFace Inference API

### Langfuse (`langfuse/`)
- `LangfuseSpanExporter` — Span export with Redis-backed buffering

### Unstructured (`unstructured/`)
- `UnstructuredPdfParser` — PDF parsing via Unstructured library

### Tika (`tika/`)
- `TikaPdfParser` — PDF parsing via Apache Tika

### Ragas (`ragas/`)
- `RagasAnswerEvaluator` — Answer quality evaluation via Ragas

### OpenTelemetry (`otel/`)
- `OtelSpanExporter` — Exports spans to an OTLP endpoint

## Security

- API keys are loaded exclusively from environment variables
- `_masked_repr()` prevents credentials from appearing in logs
- `--config` CLI overrides reject secret fields (`api_key`, `password`, `auth`)
