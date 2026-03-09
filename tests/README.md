# tests/

Deterministic test suite. All tests run locally with no external service dependencies.

## Running Tests

```bash
make test          # All tests (683 tests, ~0.3s)
make test-unit     # Unit tests only
make lint          # Ruff linting
make typecheck     # mypy strict mode
```

Or directly:

```bash
uv run pytest tests/ -v
uv run pytest tests/unit/test_query_orchestrator.py -v  # Single file
```

## Structure

```
tests/
  unit/               Deterministic unit tests
  integration/        Integration tests (planned, for real backends)
  fixtures/           Sample documents for testing
    sample.txt        Plain text fixture
    sample.md         Markdown fixture
```

## Test Infrastructure

Tests use in-memory/mock implementations exclusively:

| Mock | Replaces | Purpose |
|------|----------|---------|
| `MemoryVectorStore` | Qdrant/OpenSearch vector | Cosine similarity in a dict |
| `MemoryLexicalStore` | OpenSearch lexical | Term matching in a dict |
| `DeterministicEmbeddingProvider` | TEI | Hash-based vectors (reproducible) |
| `DeterministicQueryEmbedder` | TEI query embedding | Wraps mock provider |
| `MockGenerator` | vLLM | Deterministic answers + citations |
| `FeatureBasedReranker` | Cross-encoder | Multi-signal scoring (no model) |
| `InMemoryCollector` | OTLP exporter | Captures spans for assertion |
| `InMemoryExperimentStore` | Database | Dict-backed experiment persistence |
| `Memory*Repository` | Postgres | In-memory chunk/embedding/crawl state repos |

## Test Coverage by Subsystem

| Test File | Subsystem |
|-----------|-----------|
| `test_contracts_*.py` | Domain types (documents, chunks, retrieval, context, generation, evaluation) |
| `test_ingestion.py` | Ingestion service and change detection |
| `test_parsing.py` | Document parsers (plain text, markdown) |
| `test_chunking.py` | Chunking strategies (fixed window, recursive, parent-child) |
| `test_embeddings.py` | Embedding providers |
| `test_indexing.py` | Indexing service |
| `test_retrieval_stores.py` | Memory vector/lexical stores and adapter stubs |
| `test_retrieval_broker.py` | Retrieval broker, hybrid retrieval |
| `test_rrf_fusion.py` | RRF fusion algorithm |
| `test_broker_dedup.py` | Candidate deduplication |
| `test_reranking.py` | Reranking service and feature reranker |
| `test_context_builder.py` | Context building and token budgeting |
| `test_generation.py` | Generation pipeline, citation validation |
| `test_observability.py` | Tracer, spans, collectors |
| `test_evaluation.py` | Evaluation metrics and report generation |
| `test_experiments.py` | Experiment runner, comparison, persistence |
| `test_adapter_*.py` | Adapter config, factory, and stubs |
| `test_bootstrap.py` | ServiceRegistry creation and secret rejection |
| `test_query_orchestrator.py` | Full query pipeline with mocks |
| `test_ingestion_orchestrator.py` | Full ingestion pipeline with mocks |
| `test_cli_output.py` | Output rendering (JSON + human) |
| `test_cli_commands.py` | CLI command parsing and end-to-end execution |

## Principles

- **Deterministic:** Same inputs always produce same outputs
- **Fast:** Entire suite runs in under a second
- **Independent:** No test depends on another test's state
- **No network:** All external services replaced with in-memory mocks
