# libs/adapters/

Concrete implementations of subsystem protocols for external services, plus in-memory mocks for testing.

## Architecture

Every external dependency is behind a protocol defined in its subsystem. Adapters implement these protocols. Factory functions in `factory.py` select the right implementation based on configuration.

```
libs/adapters/factory.py    # Factory functions (create_vector_store, create_generator, ...)
libs/adapters/config.py     # Config dataclasses (QdrantConfig, VllmConfig, ...)
libs/adapters/env.py        # Environment loaders (load_qdrant_config, ...)
libs/adapters/memory/       # In-memory implementations
libs/adapters/qdrant/       # Qdrant adapter
libs/adapters/opensearch/   # OpenSearch adapter
libs/adapters/tei/          # TEI adapter
libs/adapters/vllm/         # vLLM adapter
libs/adapters/unstructured/ # Unstructured adapter
libs/adapters/tika/         # Tika adapter
libs/adapters/ragas/        # Ragas adapter
libs/adapters/otel/         # OpenTelemetry adapter
```

## Factory Functions (`factory.py`)

Each factory returns an in-memory/mock implementation when `config=None`, and the real adapter when config is provided:

| Factory | Protocol | Mock Fallback | Real Adapter |
|---------|----------|---------------|--------------|
| `create_vector_store(config)` | `VectorStore` | `MemoryVectorStore` | `QdrantVectorStore` / `OpenSearchVectorStore` |
| `create_lexical_store(config)` | `LexicalStore` | `MemoryLexicalStore` | `OpenSearchLexicalStore` |
| `create_embedding_provider(config)` | `EmbeddingProvider` | `DeterministicEmbeddingProvider` | `TeiEmbeddingProvider` |
| `create_query_embedder(config)` | `QueryEmbedder` | Mock hash-based embedder | `TeiQueryEmbedder` |
| `create_reranker(config)` | `Reranker` | `CrossEncoderReranker` (stub) | `TeiCrossEncoderReranker` |
| `create_generator(config)` | `Generator` | `MockGenerator` | `VllmGenerator` |
| `create_span_collector(config)` | `SpanCollector` | `NoOpCollector` | `OtelSpanExporter` |
| `create_pdf_parser(provider, config)` | `PdfParserBase` | — | `UnstructuredPdfParser` / `TikaPdfParser` |

## Environment Loaders (`env.py`)

Each loader reads `DRIFTER_*` environment variables and returns a config dataclass or `None` if the required vars aren't set:

| Loader | Env Vars | Config Type |
|--------|----------|-------------|
| `load_qdrant_config()` | `DRIFTER_QDRANT_*` | `QdrantConfig` |
| `load_opensearch_config()` | `DRIFTER_OPENSEARCH_*` | `OpenSearchConfig` |
| `load_tei_config()` | `DRIFTER_TEI_*` | `TeiConfig` |
| `load_vllm_config()` | `DRIFTER_VLLM_*` | `VllmConfig` |
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
| `VllmConfig` | base_url, model_id, api_key, max_tokens, temperature |
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

### vLLM (`vllm/`)
- `VllmGenerator` — LLM generation via vLLM implementing `Generator`

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
