# libs/

Core libraries for the Drifter RAG pipeline. Each subdirectory is an independent subsystem with strict boundaries.

## Subsystem Pipeline

```
ingestion -> parsing -> chunking -> embeddings -> indexing -> retrieval -> reranking -> context_builder -> generation
```

Supporting subsystems: `observability`, `evaluation`, `experiments`, `adapters`.

## Architecture Rules

- **No cross-library imports.** Subsystems communicate only through typed contracts defined in `contracts/`.
- **Protocol-based abstractions.** Every external dependency is behind a `Protocol` defined in the subsystem's `protocols.py`.
- **Each subsystem is independently testable.** All support in-memory adapter testing with deterministic behavior.
- **Observability is cross-cutting.** Every subsystem can emit spans and events via `libs/observability/`.

## Subsystems

| Subsystem | Consumes | Produces | Purpose |
|-----------|----------|----------|---------|
| `contracts/` | — | Domain types | Shared typed models (the glue between subsystems) |
| `ingestion/` | Source configs | SourceDocumentRef, RawDocument | Source discovery, crawl state, fetching |
| `parsing/` | RawDocument | CanonicalDocument | Format extraction, structure preservation |
| `chunking/` | CanonicalDocument | list[Chunk] | Document splitting with lineage |
| `embeddings/` | list[Chunk] | list[ChunkEmbedding] | Dense vector generation |
| `indexing/` | Chunks + Embeddings | IndexingResult | Vector and lexical index writes |
| `retrieval/` | RetrievalQuery | BrokerResult | Query execution, hybrid fusion |
| `reranking/` | list[RetrievalCandidate] | list[RankedCandidate] | Precision ranking |
| `context_builder/` | list[RankedCandidate] | ContextPack | Token-budgeted evidence selection |
| `generation/` | ContextPack | GeneratedAnswer | LLM reasoning with citations |
| `observability/` | (cross-cutting) | Spans, metrics | OpenTelemetry-compatible tracing |
| `evaluation/` | EvaluationCase | EvaluationReport | Retrieval quality metrics |
| `experiments/` | ExperimentConfig | ExperimentRun | Reproducible experiments |
| `adapters/` | Configs | Service instances | External service integrations |

## File Conventions

Each subsystem typically contains:

- `__init__.py` — Module docstring, `__all__` exports
- `protocols.py` — `Protocol` classes defining the subsystem's interfaces
- `models.py` — Result types, configs, enums
- `service.py` — Main service class with a `run()` method
- Additional implementation files as needed
